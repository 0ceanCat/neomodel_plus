from django.db.models.options import Options
from django_neomodel import DjangoNode, DjangoField
from neo4j.graph import Node
from neomodel import IntegerProperty as IntP, StringProperty as SP, \
	FloatProperty as FP, DateTimeProperty as DTP, DateProperty as DP, \
	ArrayProperty as AP, BooleanProperty as BP, UniqueIdProperty as UP, warnings, db
from query_builder import AdvancedNodeSet
from neomodel.util import get_node_properties
import django.db.models.options as options

from utils import load_full_text_index


def classproperty(f):
	class cpf(object):
		def __init__(self, getter):
			self.getter = getter

		def __get__(self, obj, type=None):
			return self.getter(type)

	return cpf(f)


options.DEFAULT_NAMES = options.DEFAULT_NAMES + ('default_select',)
options.DEFAULT_NAMES = options.DEFAULT_NAMES + ('composite_index',)


class RelSearchableNode(DjangoNode):
	"""
		Relationship searchable node.
		Implements methods that are able to search nodes by properties and relationships
	"""
	full_text_index = None

	@classproperty
	def _meta(self):
		if hasattr(self.Meta, 'unique_together'):
			raise NotImplementedError('unique_together property not supported by neomodel')

		opts = Options(self.Meta, app_label=self.Meta.app_label)
		opts.contribute_to_class(self, self.__name__)

		for key, prop in self.__all_properties__:
			opts.add_field(DjangoField(prop, key), getattr(prop, 'private', False))
			if getattr(prop, "primary_key", False):
				self.pk = prop
				self.pk.auto_created = True

		return opts

	@classmethod
	def full_text_search(cls, limit=100, skip=0, **kwargs):
		if cls.full_text_index is None:
			load_full_text_index(cls)

		prop_name, prop_v = kwargs.popitem()
		index_name = cls.full_text_index
		query = f'CALL db.index.fulltext.queryNodes("{index_name}", "{prop_v}") YIELD node, score ' \
		        'RETURN node, score ' \
		        f' SKIP {skip} LIMIT {limit}'
		result, meta = db.cypher_query(query)
		if result:
			return [cls.inflate(r, meta) for r in result]

	@classmethod
	def get(cls, *args, **kwargs):
		f = cls.filter(*args, **kwargs)
		return f.first()

	@classmethod
	def filter(cls, *args, **kwargs) -> AdvancedNodeSet:
		select = cls.Meta.default_select if hasattr(cls, 'Meta') and hasattr(cls.Meta, 'default_select') else None
		return AdvancedNodeSet(cls, *args, **kwargs, select=select)

	@classmethod
	def all(cls, order_by=None) -> AdvancedNodeSet:
		f = cls.filter()
		if order_by:
			f = f.order_by(order_by)
		return f

	@classmethod
	def select(cls, *args) -> AdvancedNodeSet:
		f = cls.filter()
		return f.select(*args)

	@classmethod
	def get_or_create(cls, *props, **kwargs):
		"""
		Call to MERGE with parameters map. A new instance will be created and saved if does not already exist,
		this is an atomic operation.
		Parameters must contain all required properties, any non required properties with defaults will be generated.

		Note that the post_create hook isn't called after get_or_create

		:param props: Arguments to get_or_create as tuple of dict with property names and values to get or create
					  the entities with.
		:type props: tuple
		:param relationship: Optional, relationship to get/create on when new entity is created.
		:param lazy: False by default, specify True to get nodes with id only without the parameters.
		:rtype: list
		"""
		lazy = kwargs.get('lazy', False)
		relationship = kwargs.get('relationship')

		# build merge query
		get_or_create_params = [{"create": cls.deflate(p, skip_empty=True)} for p in props]
		query, params = cls._build_merge_query(get_or_create_params, relationship=relationship, lazy=lazy)
		if len(params['merge_params']) == 0:
			params['merge_params'] = [{'create': kwargs}]
		else:
			params['merge_params'][0]['create'].update(kwargs)
		if 'streaming' in kwargs:
			warnings.warn('streaming is not supported by bolt, please remove the kwarg',
			              category=DeprecationWarning, stacklevel=1)

		# fetch and build instance for each result
		results, meta = db.cypher_query(query, params)
		return cls.inflate(results[0][0], meta)

	@classmethod
	def inflate(cls, node, meta=None):
		"""
		Inflate a raw neo4j_driver node to a neomodel node
		:param node: data returned by neo4j
		:param meta: selected properties.
		:return: node object
		"""
		# support lazy loading
		if isinstance(node, int):
			snode = cls()
			snode.id = node
		else:
			props = {}
			if not isinstance(node, Node) and (not isinstance(node, list) or not isinstance(node[0], Node)):
				meta = convert_to_property_names(meta)
				if not isinstance(node, list):
					props[meta[0]] = node
				else:
					for i in range(len(node)):
						props[meta[i]] = node[i]
				snode = cls(**props)
			else:
				if isinstance(node, list):
					node = node[0]
				node_properties = get_node_properties(node)
				for key, prop in cls.__all_properties__:
					# map property name from database to object property
					db_property = prop.db_property or key

					if db_property in node_properties:
						props[key] = prop.inflate(node_properties[db_property], node)
					elif prop.has_default:
						props[key] = prop.default_value()
					else:
						props[key] = None

				snode = cls(**props)
				snode.id = node.id

		return snode

	class Meta:
		# Used as Meta of Django
		# Default selected properties for each search
		# e.g: default_select = ('name', 'age')
		# For each search returns only selected properties, in this case they are `name` and `age`
		# If `default_select` is not declared or is empty, then all properties of the current Node will be returned
		default_select = ()

		# is equal to unique_together
		composite_index = []

	def __hash__(self):
		return hash(self.id)


class IndexableProperty:
	def __init__(self, index=False, unique_index=False, full_text_index=False, primary_key=False):
		if index and unique_index:
			raise ValueError(
				"The arguments `unique_index` and `index` are mutually exclusive."
			)

		if (index or unique_index) and isinstance(self, ArrayProperty):
			raise ValueError("Can not set index on ArrayProperty!")
		self.need_index = index
		self.unique_index = unique_index
		self.full_text_index = full_text_index
		self.primary_key = primary_key


class IntegerProperty(IntP, IndexableProperty):
	# to support creation of index on relationship property
	def __init__(self, default=0, **kwargs):
		index = kwargs.pop('index', None)
		unique_index = kwargs.pop('unique_index', None)
		primary_key = kwargs.pop('primary_key', None)
		IntP.__init__(self, default=default, **kwargs)
		IndexableProperty.__init__(self, index, unique_index, False, primary_key)


class StringProperty(SP, IndexableProperty):
	# to support creation of index on relationship property
	def __init__(self, default='', **kwargs):
		index = kwargs.pop('index', None)
		unique_index = kwargs.pop('unique_index', None)
		full_text_index = kwargs.pop('full_text_index', None)
		primary_key = kwargs.pop('primary_key', None)
		SP.__init__(self, default=default, **kwargs)
		IndexableProperty.__init__(self, index, unique_index, full_text_index, primary_key)


class FloatProperty(FP, IndexableProperty):
	# to support creation of index on relationship property
	def __init__(self, default=0.0, **kwargs):
		index = kwargs.pop('index', None)
		unique_index = kwargs.pop('unique_index', None)
		primary_key = kwargs.pop('primary_key', None)
		FP.__init__(self, default=default, **kwargs)
		IndexableProperty.__init__(self, index, unique_index, False, primary_key)


class BooleanProperty(BP, IndexableProperty):
	# to support creation of index on relationship property
	def __init__(self, default=False, **kwargs):
		index = kwargs.pop('index', None)
		unique_index = kwargs.pop('unique_index', None)
		primary_key = kwargs.pop('primary_key', None)
		BP.__init__(self, default=default, **kwargs)
		IndexableProperty.__init__(self, index, unique_index, False, primary_key)


class DateProperty(DP, IndexableProperty):
	# to support creation of index on relationship property
	def __init__(self, **kwargs):
		index = kwargs.pop('index', None)
		unique_index = kwargs.pop('unique_index', None)
		primary_key = kwargs.pop('primary_key', None)
		DP.__init__(self, **kwargs)
		IndexableProperty.__init__(self, index, unique_index, False, primary_key)


class DateTimeProperty(DTP, IndexableProperty):
	# to support creation of index on relationship property
	def __init__(self, **kwargs):
		index = kwargs.pop('index', None)
		unique_index = kwargs.pop('unique_index', None)
		primary_key = kwargs.pop('primary_key', None)
		DTP.__init__(self, **kwargs)
		IndexableProperty.__init__(self, index, unique_index, False, primary_key)


class ArrayProperty(AP, IndexableProperty):
	def __init__(self, **kwargs):
		index = kwargs.pop('index', None)
		unique_index = kwargs.pop('unique_index', None)
		primary_key = kwargs.pop('primary_key', None)
		AP.__init__(self, **kwargs)
		IndexableProperty.__init__(self, index, unique_index, False, primary_key)

	def __getitem__(self, item):
		return self.base_property[item]


class UniqueIdProperty(UP, IndexableProperty):
	def __init__(self, **kwargs):
		index = kwargs.pop('index', None)
		unique_index = kwargs.pop('unique_index', None)
		primary_key = kwargs.pop('primary_key', None)
		UP.__init__(self, **kwargs)
		IndexableProperty.__init__(self, index, unique_index, False, primary_key)


def convert_to_property_names(meta):
	properties = []

	if not meta:
		return properties
	for m in meta:
		split_m = m.split('.')
		if len(split_m) == 2:
			# e.g: n.name
			properties.append(split_m[1])
		else:
			# ID(n)
			properties.append('id')
	return properties
