import datetime
import inspect
from collections import defaultdict
from io import StringIO
from typing import TypeVar

import django
from django.db.models import Aggregate

from neomodel import StructuredNode, Q
from neomodel.properties import Property
from utils import str_to_class, db, is_collection
from neomodel.relationship_manager import RelationshipDefinition


class SearchType:
    REL_PROP = 1  # relationship property
    RECUR_PROP = 2  # recursively process property filter
    RECUR_NSET = 3  # recursively process the node set
    ID = 4  # id
    PROP = 5  # node property


class KeywordArgument:
    def __init__(self, negated, parameter, operator, value, type=SearchType.PROP):
        self.negated = negated
        self.parameter = parameter
        self.operator = operator
        self.value = value
        self.type = type

    def __iter__(self):
        yield self.negated
        yield self.parameter
        yield self.operator
        yield self.value
        yield self.type

    def __len__(self):
        return 4

    def __str__(self):
        return f'[negated: {self.negated}, parameter: {self.parameter}, operator: {self.operator},' \
               f' value: {self.value.__class__ if isinstance(self.value, AdvancedNodeSet) else self.value}]'

    def __repr__(self):
        return str(self)


class RelKeywordArgument:
    def __init__(self, rel_path, keyword_argument):
        self.rel_path = rel_path
        self.keyword_argument = keyword_argument

    def __iter__(self):
        yield self.rel_path
        yield self.keyword_argument

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'[rel_path: {self.rel_path}, keyword_argument: {self.keyword_argument}]'


class RelPath:
    def __init__(self, node_name, rel_type, rel_direction, cls):
        self.node_name = node_name
        self.rel_type = rel_type
        self.rel_direction = rel_direction
        self.cls = cls

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'[node_name: {self.node_name}, rel_type: {self.rel_type}, rel_direction: {self.rel_direction}, cls: {self.cls.__name__}]'


class NodeRelationshipDetail:
    def __init__(self, cls):
        self.cls = cls
        self.relationships = {k: v for k, v in vars(cls).items() if
                              isinstance(v, RelationshipDefinition)}

        self.linked_nodes = {v._raw_class.lower(): str_to_class(v._raw_class) for k, v in self.relationships.items()}
        self.node_properties = {k for k, v in vars(cls).items() if isinstance(v, Property)}
        self.linked_nodes[self.cls_name.lower()] = self.cls

    @property
    def cls_name(self):
        return self.cls.__name__

    def is_node_property(self, property_name):
        return property_name.lower() in self.node_properties

    def is_node_relationship(self, name):
        return name in self.relationships

    def is_connected_with(self, node):
        if inspect.isclass(node) and issubclass(node, StructuredNode):
            node_name = node.__name__
        else:
            node_name = node
        return node_name.lower() in self.linked_nodes

    def get_linkedNode_cls(self, node_name):
        try:
            return self.linked_nodes[node_name.lower()]
        except KeyError:
            raise KeyError(f"Node {self.cls_name} has no relationship with {node_name.capitalize()}")

    def get_rel_detail(self, name=None, inverse=False) -> RelPath:
        """
        get the connected node's class name and the relationship type with it
        e.g:  image  -- belong to --> user
        Image.get_rel_detail('user') -> ConnectedNode('User', 'belong_to', direction, User.__class__)
        """

        rel = None
        try:
            rel = self.relationships[name]
        except KeyError:
            for v in self.relationships.values():
                if v._raw_class == name:
                    rel = v

            if rel is None:
                raise KeyError(f"Node {self.cls_name} has no relationship with {name.capitalize()}")

        rel_type = rel.definition['relation_type']
        direction = rel.definition['direction']
        connected_node_name = rel._raw_class

        if inverse:
            return RelPath(self.cls_name, rel_type, -direction, self.cls)
        else:
            return RelPath(connected_node_name, rel_type, direction, self.get_linkedNode_cls(connected_node_name))


class _OPERATORS_CLASS:
    _OPERATORS = {'gt': '>', 'gte': '>=', 'ne': '<>', 'lt': '<', 'lte': '<=',
                  'startswith': 'STARTS WITH', 'endswith': 'ENDS WITH', 'in': 'IN',
                  'contains': 'CONTAINS', 'eq': '='}

    GT = _OPERATORS['gt']
    GTE = _OPERATORS['gte']
    NE = _OPERATORS['ne']
    LT = _OPERATORS['lt']
    LTE = _OPERATORS['lte']
    STARTSWITH = _OPERATORS['startswith']
    ENDSWITH = _OPERATORS['endswith']
    IN = _OPERATORS['in']
    CONTAINS = _OPERATORS['contains']
    EQ = _OPERATORS['eq']

    def __getitem__(self, key):
        operator = self._OPERATORS.get(key, None)
        if operator is None:
            raise KeyError(f"There is no operator named `{key}`")
        return operator

    def __contains__(self, key):
        return self._OPERATORS.get(key, None) is not None


class CypherKeyWords:
    class KeyWord:
        def __init__(self, name):
            self.name = name

        def __eq__(self, other):
            if isinstance(other, str):
                return other.lower() == self.name.lower()

            elif isinstance(other, CypherKeyWords.KeyWord):
                return other.name == self.name

            return False

        def __repr__(self):
            return self.name

        def __add__(self, other):
            return self.name + other

    RETURN = KeyWord(name="RETURN")
    MATCH = KeyWord(name="MATCH")
    WHERE = KeyWord(name="WHERE")
    WITH = KeyWord(name="WITH")
    AND = KeyWord(name="AND")
    NOT = KeyWord(name="NOT")
    OR = KeyWord(name="OR")
    UNION = KeyWord(name="UNION")
    DISTINCT = KeyWord(name="DISTINCT")
    ORDER_BY = KeyWord(name="ORDER BY")
    SKIP = KeyWord(name="SKIP")
    LIMIT = KeyWord(name="LIMIT")


class Variable:
    def __init__(self, newly_created, name):
        self.newly_created = newly_created
        self.name = name


OPERATORS = _OPERATORS_CLASS()


def remove_duplication(params):
    r = []
    for i in params:
        if i not in r:
            r.append(i)
    return r


class AttributeRelConditions:
    class RelConditions:
        def __init__(self, ):
            self.and_rel_conditions = defaultdict(list)
            self.or_rel_conditions = defaultdict(list)

        def __bool__(self):
            return len(self.and_rel_conditions) + len(self.or_rel_conditions) > 0

        def gree(self):
            return int(bool(self.and_rel_conditions)) + int(bool(self.or_rel_conditions))

        def add(self, key, value, AND=True):
            if AND:
                self.and_rel_conditions[key].append(value)
            else:
                self.or_rel_conditions[key].append(value)

        def extend(self, key, list_, AND=True):
            if AND:
                self.and_rel_conditions[key].extend(list_)
            else:
                self.or_rel_conditions[key].extend(list_)

        def merge(self, rel_conditions):
            def _merge(dictA, dictB):
                for k in dictB.keys():
                    dictA[k].extend(dictB[k])

            _merge(self.and_rel_conditions, rel_conditions.and_rel_conditions)
            _merge(self.or_rel_conditions, rel_conditions.or_rel_conditions)

    class AttributeConditions:
        def __init__(self):
            self.and_attr_conditions = []
            self.or_attr_conditions = []

        def __getitem__(self, item):
            if item == CypherKeyWords.AND:
                return self.and_attr_conditions
            return self.or_attr_conditions

        def __bool__(self):
            return len(self.and_attr_conditions) + len(self.or_attr_conditions) > 0

        def gree(self):
            return int(bool(self.and_attr_conditions)) + int(bool(self.or_attr_conditions))

        def add(self, value, AND=True):
            if AND:
                self.and_attr_conditions.append(value)
            else:
                self.or_attr_conditions.append(value)

        def merge(self, attr_conditions):
            self.and_attr_conditions.extend(attr_conditions.and_attr_conditions)
            self.or_attr_conditions.extend(attr_conditions.or_attr_conditions)

    def __init__(self):
        # attribute conditions for the current Node
        self.rel_conditions = AttributeRelConditions.RelConditions()

        # relationships directly used by the current Node
        self.attr_conditions = AttributeRelConditions.AttributeConditions()

    def merge(self, conditions):
        self.rel_conditions.merge(conditions.rel_conditions)
        self.attr_conditions.merge(conditions.attr_conditions)

    def add_attr_condition(self, condition, AND=True):
        self.attr_conditions.add(condition, AND)

    def add_rel_condition(self, key, condition, AND=True):
        self.rel_conditions.add(key, condition, AND)


class QueryParser:
    def __init__(self, node_cls_detail):
        self.node_cls_detail = node_cls_detail

    def parse_param_list(self, connector, param_list, negated=False):
        conditions = AttributeRelConditions()
        param_list = remove_duplication(param_list)
        is_AND = connector == CypherKeyWords.AND
        for param_v in param_list:
            if isinstance(param_v, Q):
                conditions.merge(self.parse_param_list(param_v.connector, param_v.children, param_v.negated))
                continue

            param, v = param_v  # e.g: Q(user__username='root') ->  param = 'user__username', v = 'root'
            if is_collection(v) and not isinstance(v, AdvancedNodeSet):
                v = list(v)
            split_param = param.split("__")
            operator = OPERATORS.EQ  # default operator

            if self.node_cls_detail.is_node_property(split_param[0]):
                param = split_param[0]

                if len(split_param) > 1:
                    # if it is a query like age__gte=10
                    # convert 'gte' to operator '>='
                    operator = OPERATORS[split_param[1]]

                conditions.add_attr_condition(KeywordArgument(negated, param, operator, v),
                                              is_AND)
            elif split_param[0] in ('in_', 'in'):
                # e.g: Image.filter(in_=[list of images or list of ids])
                conditions.add_attr_condition(KeywordArgument(negated, None, OPERATORS.IN, v, type=SearchType.ID),
                                              is_AND)
            elif split_param[0] == 'id':
                # e.g: Image.filter(id=1)
                conditions.add_attr_condition(KeywordArgument(negated, None, operator, v, type=SearchType.ID),
                                              is_AND)
            elif split_param[0].startswith('rel_'):
                # search based on relationship property
                if len(split_param) > 2:
                    # e.g: rel_tag__quantity__gte=10.
                    # The relationship that connects the current node and Tag node has property `quantity` >= 10
                    operator = OPERATORS[split_param[2]]

                conditions.add_rel_condition(split_param[0][4:], KeywordArgument(negated, split_param[1], operator, v,
                                                                                 type=SearchType.REL_PROP), is_AND)
            else:
                # relationship search
                if len(split_param) == 1:
                    if isinstance(v, AdvancedNodeSet):
                        # e.g: Image.filter(tag=Tag.filter(name='a'))
                        # where `tag` is a declared property in Image class that represents the relationship between Image and Tag
                        conditions.add_rel_condition(split_param[0],
                                                     KeywordArgument(negated, None, operator, v,
                                                                     type=SearchType.RECUR_NSET), is_AND)

                    else:
                        # e.g: Image.filter(tag=Tag())
                        conditions.add_rel_condition(split_param[0],
                                                     KeywordArgument(negated, None, operator, v, type=SearchType.ID),
                                                     is_AND)

                else:
                    # e.g: Image.filter(tag__name__startswith='a')
                    conditions.add_rel_condition(split_param[0],
                                                 KeywordArgument(negated, f"{'__'.join(split_param[1:])}", operator, v,
                                                                 type=SearchType.RECUR_PROP), is_AND)
        return conditions


class QueryBuilderResult:
    def __init__(self, cypher, returned_var, type_counter):
        self.cypher = cypher
        self.returned_var = returned_var
        self.type_counter = type_counter


def _space(space):
    return " " if space else ""


class QueryBuilder:
    VAR_COUNTER = 0
    NODE_DETAILS = {}

    def __init__(self, advanced_filter):
        self.advanced_filter = advanced_filter
        self.node_cls_detail = self._get_node_detail(advanced_filter.cls)
        self.variable_table = defaultdict(str)
        self.type_counter = defaultdict(int)
        self.query_builder = StringIO()
        self.parser = QueryParser(self.node_cls_detail)

    def _build_query(self, without_return=False, counter=0):
        self.counter = counter
        queries = self.advanced_filter._filter
        cls_name = self.node_cls_detail.cls_name
        if len(queries.children) == 0:
            # when user passes an empty Q, e.g: Image.filter(Q())
            variable = self._get_var(cls_name)
            self._write(f"{CypherKeyWords.MATCH} ({variable.name}:{cls_name}) ")
        else:

            # e.g: query_1 = Q(age=1), query_2 = Q(name_startswith='d')
            # c = query_1 & query_2. In this case, the connector between the queries is 'AND'
            # c = query_1 | query_2. The connector between the queries is 'OR'
            # if convert these queries to cypher, they will be a combination of MATCH queries
            inter_query_connector = queries.connector
            children = [e for e in queries.children if not isinstance(e, tuple)]
            children.extend([e for e in queries.children if isinstance(e, tuple)])
            children = remove_duplication(children)
            for i, query in enumerate(children):
                # e.g: query_1 = Q(age=1) | Q(name_startswith='d')
                # if convert to cypher it is a combination of filters: WHERE age=1 OR name STARTS WITH 'd'
                inter_param_connector = CypherKeyWords.AND  # default connector
                if isinstance(query, Q):
                    inter_param_connector = query.connector
                    param_list = query.children
                    negated = query.negated
                else:
                    # if the last query has only 0 or 1 param, it is a tuple object
                    # so, it should be converted to a list
                    if isinstance(query, tuple) or isinstance(query, AdvancedNodeSet):
                        query = [query]

                    param_list = query
                    negated = False

                if len(param_list) == 0:
                    # when user passes an empty Q, e.g: Image.filter(Q())
                    variable = self._get_var(cls_name)
                    self._write_match_stmt(variable.name, cls_name)
                else:
                    conditions = self.parser.parse_param_list(inter_param_connector, param_list,
                                                              negated)
                    self._make_basic_cypher(conditions.attr_conditions)

                    self._make_cypher_for_rels(conditions.rel_conditions)

                    self._add_merge_key_word(i, len(queries.children), inter_query_connector)
                    self._new_line()

        self._terminal_stmt(without_return)
        result = QueryBuilderResult(self.query_builder.getvalue(), self._get_var(cls_name).name,
                                    self._get_type_counter(self.node_cls_detail.cls))
        return result

    def _terminal_stmt(self, without_return):
        cls_name = self.node_cls_detail.cls_name

        aggregate = self.advanced_filter._aggregate
        if not without_return:
            variable = self._get_var(cls_name)
            count_records = self.advanced_filter._count_records
            orderby_params = self.advanced_filter._order_by

            if not aggregate and not count_records:
                orderby_params = self._create_orderby_params(variable.name, orderby_params)

            self._write_return_stmt(variable.name, aggregate, count_records)

            if not aggregate:
                self._write_order_by_stmt(orderby_params)

                self._write_skip_stmt()

                self._write_limit_stmt()

    def _create_orderby_params(self, cls_var, odby):
        orderby_params = []

        # verify if order_by properties belong to the current node or to some node directly connected with it
        for o in odby:
            prop, order = o
            is_current_node_property = self._check_if_current_node_property(prop)

            # check if is a property of a directly connected node
            is_directly_connected_node_property = self._check_if_connected_node_property(self.node_cls_detail.cls, prop)

            if not is_current_node_property and not is_directly_connected_node_property and prop != 'id':
                raise AttributeError(
                    f"{self.node_cls_detail.cls_name} does not have property '{prop}'!")

            orderby_params.append((cls_var, prop, order))

        return orderby_params

    def _check_if_current_node_property(self, prop_name):
        return self.node_cls_detail.is_node_property(prop_name)

    def _check_if_connected_node_property(self, clzz, prop_name):
        self.node_cls_detail.get_linkedNode_cls(clzz.__name__)
        directly_connected_node = self._get_node_detail(clzz)
        return directly_connected_node.is_node_property(prop_name)

    def _change_if_v_not_adequate(self, v):
        if isinstance(v, str):
            return f"'{v}'"
        if isinstance(v, datetime.datetime):
            return v.timestamp()
        elif isinstance(v, datetime.date):
            return f"'{str(v)}'"
        return v

    def _add_merge_key_word(self, i: int, list_len: int, connector: CypherKeyWords.KeyWord,
                            condition_filter: bool = False):
        cls = self.advanced_filter.cls
        if i + 1 < list_len:
            # if there are more queries, add the keyword UNION if the connector is OR
            if condition_filter:
                self._write(connector, space=True)
            elif connector == CypherKeyWords.OR:
                self._new_line()
                self._write(f'{CypherKeyWords.RETURN} {self._get_var(cls.__name__).name}')
                self._new_line()
                self._write(CypherKeyWords.UNION)
                self._new_line()
                self._clear_variable_table()
            elif connector == CypherKeyWords.AND:
                self._new_line()
                self._write(f"{CypherKeyWords.WITH} {self._get_var(cls.__name__).name} ")
                self._clear_variable_table(False)

    def _make_basic_cypher(self, attr_conditions: AttributeRelConditions.AttributeConditions):
        """
        e.g: MATCH (n:Node)
        """
        if not attr_conditions:
            return

        cls_name = self.node_cls_detail.cls_name
        variable = self._get_var(cls_name)

        # if the current node type was never used, declare a new variable
        # ex: MATCH (varA: Image)
        # if the current node type has been used, use the declared variable
        # ex: MATCH (varA)
        self._write_match_stmt(variable.name, cls_name if not variable.newly_created else None)

        self._write_where_stmt(self.node_cls_detail.cls, attr_conditions.and_attr_conditions, CypherKeyWords.AND,
                               variable.name, False)

        self._add_merge_key_word(0, attr_conditions.gree(), CypherKeyWords.AND, condition_filter=True)

        self._write_where_stmt(self.node_cls_detail.cls, attr_conditions.or_attr_conditions, CypherKeyWords.OR,
                               variable.name, True)

    def _write_property_filters(self, condition_list: list, connector: CypherKeyWords.KeyWord, target_type: type,
                                var: str):
        for j, condition in enumerate(condition_list):
            if isinstance(condition, RelKeywordArgument):
                rel_path, condition = condition

            negated, param, operator, v, type = condition
            not_ = CypherKeyWords.NOT if negated else ''

            if type == SearchType.ID:
                if is_collection(v):
                    self._write_id_stmt(not_, var, operator, self._convert_nodes_to_ids(target_type, v))
                else:
                    self._check_type(v, target_type)
                    v = self._to_id(v)
                    self._write_id_stmt(not_, var, OPERATORS.EQ, v)

            elif type == SearchType.REL_PROP:
                rel_type = rel_path.rel_type
                self._write_value_stmt(not_, self._get_var(rel_type).name, param, operator,
                                       self._change_if_v_not_adequate(v))
            else:
                if operator == OPERATORS.IN and not isinstance(v, list):
                    v = list(v) if isinstance(v, tuple) else [v]

                self._write_value_stmt(not_, var, param, operator, self._change_if_v_not_adequate(v))

            self._add_merge_key_word(j, len(condition_list), connector, condition_filter=True)

    def _make_rel_cypher(self, rel_node_name, rel_type, rel_direction, continuation=False):
        """
        make a relationship path with the current node
        and return the used variable
        e.g: `MATCH (a)-[b]-(c)`
        """

        connected_node_variable = self._get_var(rel_node_name)
        connected_node_var = connected_node_variable.name
        to = rel_direction == 1
        left = right = '-'
        if to:
            right = '->'
        else:
            left = '<-'

        rel_variable = self._get_var(rel_type)
        rel_var = rel_variable.name

        if continuation:
            if connected_node_variable.newly_created:
                self._write_basic_relationship_stmt(left, rel_var, rel_type, right, connected_node_var)
            else:
                self._write_basic_relationship_stmt(left, rel_var, rel_type, right, connected_node_var, rel_node_name)
        else:
            current_node_variable = self._get_var(self.node_cls_detail.cls_name)
            current_node_var = current_node_variable.name

            if current_node_variable.newly_created:
                # if this variable has already been created
                self._write_match_stmt(current_node_var, space=False)
            else:
                self._write_match_stmt(current_node_var, self.node_cls_detail.cls_name, space=False)

            if connected_node_variable.newly_created:
                # if this variable has already been created
                self._write_basic_relationship_stmt(left, rel_var, rel_type, right, connected_node_var)
            else:
                self._write_basic_relationship_stmt(left, rel_var, rel_type, right, connected_node_var, rel_node_name)

        return connected_node_var

    def _make_cypher_for_rels(self, rel_conditions):
        self._write_relationship_stmt(rel_conditions.and_rel_conditions, connector=CypherKeyWords.AND)
        self._add_merge_key_word(0, rel_conditions.gree(), CypherKeyWords.AND)
        self._write_relationship_stmt(rel_conditions.or_rel_conditions, connector=CypherKeyWords.OR)

    def _write_relationship_stmt(self, condition_list: dict, connector):
        rel_names = sorted(list(condition_list.keys()))
        len_rel_names = len(rel_names)

        for j in range(0, len_rel_names):
            rel = rel_names[j]

            self._convert_to_arguments(condition_list, rel)

            filters = []

            # process first the AdvancedNodeSet filters
            for i, rel_keyword_argument in enumerate(condition_list[rel]):
                rel_path: RelPath = rel_keyword_argument.rel_path
                keyword_argument: KeywordArgument = rel_keyword_argument.keyword_argument

                if isinstance(keyword_argument.value, AdvancedNodeSet):
                    # parse the AdvancedNodeSet recursively
                    rel_var = self._merge_additional_filter(keyword_argument.value)
                    rel_path.node_name = f'{rel_path.node_name}_{self._get_type_counter(rel_path.cls)}'
                    self._update_variable_table(rel_path.node_name, rel_var)
                    filters.append(rel_keyword_argument)

                self._make_rel_cypher(rel_path.node_name, rel_path.rel_type, rel_path.rel_direction)

                self._add_merge_key_word(i, len(condition_list[rel]), connector)

            # delete processed filters from the list
            condition_list[rel] = [i for i in condition_list[rel] if i not in filters]

            # then process the normal attribute filters
            for rel_keyword_argument in condition_list[rel]:
                rel_path: RelPath = rel_keyword_argument.rel_path
                keyword_argument: KeywordArgument = rel_keyword_argument.keyword_argument

                self._write_where_stmt(rel_path.cls, [RelKeywordArgument(rel_path, keyword_argument)], connector,
                                       self._get_var(rel_path.node_name).name)
                self._add_merge_key_word(j + 1, len_rel_names, connector)

    def _convert_to_arguments(self, condition_list, rel):
        for i in range(len(condition_list[rel])):
            kargs = condition_list[rel][i]
            param, v, type = kargs.parameter, kargs.value, kargs.type
            if type != SearchType.ID and not self.node_cls_detail.is_node_relationship(rel):
                if type == SearchType.RECUR_NSET:
                    # There is no defined relationship between Image and Tag in the class Image
                    # But we have the relationship declared in the class Tag
                    # So we can do something like: Image.filter(tag=Tag.filter(name='abc'))
                    detail = self._get_node_detail(v.cls)
                    if detail.is_connected_with(self.node_cls_detail.cls):
                        rel_path = detail.get_rel_detail(self.node_cls_detail.cls_name, inverse=True)
                        condition_list[rel][i] = RelKeywordArgument(rel_path, kargs)
                    else:
                        raise ValueError(
                            f"There is no connection between {self.node_cls_detail.cls_name} and {v.cls.__name__}")
                else:
                    # e.g: Image.filter(tag__name='a')
                    # e.g: Image.filter(tag=Q(name='a'))
                    # Note that, Image does not have any property called `tag`,
                    # but we know that the relationship between Image and tag is declared in the class Tag.
                    # So we can use Tag class' name as a property.
                    try:
                        connected_class = str_to_class(rel)
                        detail = self._get_node_detail(connected_class)
                        if detail.is_connected_with(self.node_cls_detail.cls):
                            rel_path = detail.get_rel_detail(self.node_cls_detail.cls_name, inverse=True)
                            if type in (SearchType.ID, SearchType.REL_PROP):
                                # e.g: Image.filter(tag=Tag())
                                condition_list[rel][i] = RelKeywordArgument(rel_path,
                                                                            KeywordArgument(*condition_list[rel][i]))
                            else:
                                # type == SearchType.RECUR_PROP
                                # e.g: Image.filter(tag__name='a')
                                condition_list[rel][i] = RelKeywordArgument(rel_path,
                                                                            KeywordArgument(kargs.negated,
                                                                                            kargs.parameter,
                                                                                            kargs.operator,
                                                                                            detail.cls.filter(
                                                                                                **{param: v})))
                        else:
                            self._no_property_error(rel)
                    except KeyError:
                        self._no_property_error(rel)

            else:
                rel_path = self.node_cls_detail.get_rel_detail(rel)
                if type == SearchType.RECUR_PROP:
                    # need to be recursively processed
                    condition_list[rel][i] = RelKeywordArgument(rel_path,
                                                                KeywordArgument(kargs.negated, kargs.parameter,
                                                                                kargs.operator,
                                                                                rel_path.cls.filter(**{param: v})))
                else:
                    # type == SearchType.RECUR_NSET
                    condition_list[rel][i] = RelKeywordArgument(rel_path,
                                                                KeywordArgument(*condition_list[rel][i]))

    def _transform_select(self, select):
        for i, prop in enumerate(select):
            if not self._check_if_current_node_property(prop):
                if prop == 'id':
                    select[i] = (True, prop)
                else:
                    raise AttributeError(
                        f"{self.node_cls_detail.cls_name} does not have property '{prop}'!")
            else:
                select[i] = (False, prop)

    def _write(self, s: str, space=False):
        self.query_builder.write(s + _space(space))

    def _new_line(self):
        return self._write("\n")

    def _write_basic_relationship_stmt(self, left: str, rel_var: str, rel_type: str, right: str, node_var: str,
                                       node_type: str = None):
        if node_type:
            self._write(f"{left}[{rel_var}:`{rel_type}`]{right}({node_var}:{node_type}) ")
        else:
            self._write(f"{left}[{rel_var}:`{rel_type}`]{right}({node_var}) ")

    def _write_match_stmt(self, var: str, clazz=None, space=True):
        if clazz is None:
            self._write(f"{CypherKeyWords.MATCH} ({var}){_space(space)}")
        else:
            self._write(f'{CypherKeyWords.MATCH} ({var}:{clazz}){_space(space)}')

    def _write_id_stmt(self, negated: str, variable_name: str, operator: str, value: int):
        self._write(f"{negated} ID({variable_name}) {operator} {value} ")

    def _write_value_stmt(self, negated: str, variable_name: str, param: str, operator: str, value):
        self._write(f"{negated} {variable_name}.{param} {operator} {value} ")

    def _write_limit_stmt(self):
        lmt = self.advanced_filter._limit
        if lmt:
            self._write(f"{CypherKeyWords.LIMIT} {lmt} ")

    def _write_where_stmt(self, target_type: type, condition_list: list, connector: CypherKeyWords.KeyWord, var: str,
                          continuation: bool = False):
        """
        Write where statement for the current MATCH query
        """
        if not continuation:
            self._write(CypherKeyWords.WHERE, space=True)

        if connector == CypherKeyWords.OR and len(condition_list) > 1:
            self._write('(')

        self._write_property_filters(condition_list, connector, target_type, var)

        if connector == CypherKeyWords.OR and len(condition_list) > 1:
            self._write(') ')

    def _write_skip_stmt(self):
        skp = self.advanced_filter._skip
        if skp:
            self._write(f"{CypherKeyWords.SKIP} {skp} ")

    def _write_order_by_stmt(self, orderby_params):
        if orderby_params:
            orderby_stmt = ', '.join([f"{o[0]}.{o[1]} {o[2]}" for o in orderby_params])
            self._write(f" {CypherKeyWords.ORDER_BY} {orderby_stmt}")

    def _write_return_stmt(self, variable: str, aggregate: str, count=False):
        if aggregate:
            self._write(f'{CypherKeyWords.RETURN} {aggregate.name}({variable}.{aggregate.identity[1][1][0]})')
            return

        select = []
        if self.advanced_filter._select:
            select = self.advanced_filter._select
            self._transform_select(select)

        self._write_distinct_stmt(select, variable)

        self._write(CypherKeyWords.RETURN, space=True)

        if count:
            self._write(f' COUNT({variable})')
        else:
            if select:
                self._write_select_stmt(select, variable)
            else:
                self._write(variable)

    def _write_select_stmt(self, select: list, variable: str):
        for i, prop in enumerate(select):
            id_prop, prop = prop
            if i > 0:
                self._write(", ")

            if id_prop:
                self._write(f"ID({variable}) ")
            else:
                self._write(f"{variable}.{prop} ")

    def _write_distinct_stmt(self, select: list, variable: str):
        if self.advanced_filter._distinct:
            distinct_stmt = ','.join([f"{variable}.{prop}" for is_id, prop in select if not is_id])
            self._write(f"{CypherKeyWords.WITH} {CypherKeyWords.DISTINCT} {distinct_stmt} ")

    def _no_property_error(self, rel):
        raise ValueError(
            f"There is no property or connected Node called {rel} in {self.node_cls_detail.cls_name}")

    def _merge_additional_filter(self, additional_filter):
        """
        make cypher for an additional filter and merge it with the current cypher
        """

        # a counter for each node type, because a node can have a relationship that points to itself
        # e.g: MATCH (image_1:Image)-[:points_to_another_image]->(image_2:Image)
        c = self._update_type_counter(type=additional_filter.cls)
        result = QueryBuilder(additional_filter) \
            ._build_query(without_return=True, counter=c)

        self._update_type_counter(type=additional_filter.cls, newV=result.type_counter)

        self._write(result.cypher)
        return result.returned_var

    def _get_type_counter(self, type):
        c = self.type_counter[type]
        if c == 0:
            c = self.counter
        self.type_counter[type] = c
        return c

    def _update_type_counter(self, type, newV=None):
        if newV:
            self.type_counter[type] = newV
        else:
            c = self.type_counter[type]
            if c == 0:
                c = self.counter
            c += 1
            self.type_counter[type] = c
            return c

    def _get_var(self, key):
        """
        create a variable for the given key
        """
        variable = self.variable_table.get(key)
        exists = True
        if not variable:
            exists = False
            variable = f"var_{key.lower()}_{self.counter}"
            self._update_variable_table(key, variable)
        return Variable(exists, variable)

    def _update_variable_table(self, cls_name, var_name):
        self.variable_table[cls_name] = var_name

    def _clear_variable_table(self, clear_all=True):
        if clear_all:
            self.variable_table.clear()
        else:
            self.variable_table = {self.node_cls_detail.cls_name: self.variable_table[self.node_cls_detail.cls_name]}

    @classmethod
    def _convert_nodes_to_ids(cls, target_type, nodes):
        """
        convert a list of nodes to a list of ids
        """
        check = cls._check_type

        if is_collection(nodes):
            if nodes:
                return [cls._to_id(e) for e in nodes if check(e, target_type)]
            raise ValueError("Empty collection detected")
        return nodes

    @staticmethod
    def _to_id(e):
        if isinstance(e, int):
            return e
        else:
            return e.id

    @staticmethod
    def _check_type(given_obj, target_type):
        """
        check type of given object
        """
        if not isinstance(given_obj, StructuredNode) and \
                not isinstance(given_obj, target_type) and \
                not isinstance(given_obj, int):
            raise ValueError(f"Invalid ID values!")

        return True

    @staticmethod
    def _get_node_detail(clazz: object):
        details = QueryBuilder.NODE_DETAILS.get(clazz.__name__)
        if details is None:
            details = NodeRelationshipDetail(clazz)
            QueryBuilder.NODE_DETAILS[clazz.__name__] = details
        return details


class AdvancedNodeSet:

    def __init__(self, cls, *args, **kwargs):
        self._filter = Q()
        self.cls = cls
        self.query_cls = QueryBuilder
        self._cypher_query = ''
        self._exclude = Q()
        self._skip = 0
        self._limit = 0
        self._distinct = False
        self._order_by = []
        self._cache = []
        self._select = []
        self._full_text = []
        self._aggregate = None
        self._count_records = False
        self._default_select = kwargs.pop('select', None)
        self._parse(*args, **kwargs)

    def _parse(self, *args, **kwargs):
        """
        Apply filters to the existing nodes in the set.
        Supports advanced relationship filter.

        :param kwargs: filter parameters

            Filters mimic Django's syntax with the double '__' to separate field and operators.

            e.g `.adv_filter(salary__gt=20000)` results in `salary > 20000`.

            Advanced relationship filter:
            e.g `.`adv_filter(link__sender__name__startswith='d')` results in MATCH (n1)-[r1]-(s:Sender) WHERE s.name STARTS WITH 'd' RETURN n1.
                    where n1 will be replaced by the tag of the node who created the current AdvancedFilter object,
                    and r1 will be replaced by the relationship between the n1 and Sender

            e.g `.`adv_filter(link__sender__link__provider__name__startswith='d')` results in
                (n1)-[r1]-(s:Sender)-[r2]-(p:Provider) WHERE p.name STARTS WITH 'd'
                n1, r1, r2 will be replaced according to the property of the node who created the current AdvancedFilter object

            e.g if you want to get Image nodes which have Tag `cat` with quantity >= 10 you can do:
                Image.adv_filter(link__tag__name='cat', rel_tag__quantity__gte=10),
                Relationship property filter MUST be used together with a Node filter even it is an empty filter!

            The following operators are available:

             * 'lt': less than
             * 'gt': greater than
             * 'lte': less than or equal to
             * 'gte': greater than or equal to
             * 'in' or 'in_': matches one of list (or tuple), if you want filter relationship by 'in', you must pass a list of Nodes
             * 'contains': contains string
             * 'startswith': string starts with
             * 'endswith': string ends with

        :return: self
        """

        other_filter, queries = self._parse_param_list(args)
        if self._filter:
            clone = self._clone()
            for i in other_filter:
                clone &= i
            clone &= Q(*queries, **kwargs)
            return clone
        else:
            for i in other_filter:
                self &= i
            self &= Q(*queries, **kwargs)
            return self

    @property
    def cypher_query(self):
        return self.query_cls(self)._build_query(False).cypher

    def aggregate(self, func: Aggregate):
        clone = self._clone()
        clone._aggregate = func
        return db.cypher_query(clone.cypher_query)[0][0][0]

    def full_test_search(self, **kwargs):
        clone = self._clone()
        prop_name, prop_v = kwargs.popitem()
        clone._full_text.append((prop_name, prop_v, self.cls.full_text_index[prop_name]))
        return clone

    def filter(self, *args, **kwargs):
        return self.intersection(*args, **kwargs)

    def distinct(self):
        clone = self._clone()
        clone._distinct = True
        return clone

    def order_by(self, *keys):
        clone = self._clone()
        order_by = []
        for key in list(keys):
            if key[0] == '-':
                order_by.append((key[1:], 'DESC'))
            elif key[0] == '+':
                order_by.append((key[1:], 'ASC'))
            else:
                order_by.append((key, 'ASC'))
        clone._order_by = order_by
        return clone

    def select(self, *args):
        clone = self._clone()
        clone._select = list(args)
        return clone

    def limit(self, limit: int):
        self._check_bound(limit, 'limit')
        clone = self._clone()
        clone._limit = limit
        return clone

    def exclude(self, *args, **kwargs):
        clone = self._clone()
        if clone._filter:
            clone._filter &= ~Q(*args, **kwargs)
        else:
            clone._filter = ~Q(*args, **kwargs)
        return clone

    def skip(self, skip: int):
        self._check_bound(skip, 'skip')
        clone = self._clone()
        clone._skip = skip
        return clone

    def union(self, *args, **kwargs):
        clone = self._clone()
        other_filter, queries = self._parse_param_list(args)
        for of in other_filter:
            clone |= of
        clone |= Q(*queries, **kwargs)
        return clone

    def intersection(self, *args, **kwargs):
        clone = self._clone()
        other_filter, queries = self._parse_param_list(args)
        for of in other_filter:
            clone &= of
        clone &= Q(*queries, **kwargs)
        return clone

    def count(self):
        if self._cache:
            return len(self._cache)
        else:
            clone = self._clone()
            clone._count_records = True
            result, _ = db.cypher_query(clone.cypher_query)
            if result:
                return result[0][0]
        return 0

    def first(self):
        if self._cache:
            return self._cache[0]
        else:
            f = self.limit(1)
            return f.fetch_all()

    def last(self):
        if self._cache:
            return self._cache[-1]
        else:
            f = self.limit(1)
            new_orderby = []
            for o in f._order_by:
                clzz, prop, order = o
                order = 'DESC' if order == 'ASC' else 'ASC'
                new_orderby.append((clzz, prop, order))

            f._order_by = new_orderby

            if not f._order_by:
                f = f.order_by('-id')

            return f.fetch_all()

    def fetch_one(self):
        return self.first()

    def fetch_all(self):
        if not self._cache:
            result, meta = db.cypher_query(self.cypher_query)
            cls = self.cls
            self._cache = [cls.inflate(r, meta) for r in result]
        return self._cache

    def _clone(self):
        clone = AdvancedNodeSet(self.cls)
        for prop in vars(clone):
            setattr(clone, prop, getattr(self, prop))
        clone._cypher_query = ''
        return clone

    def _check_bound(self, n, metd):
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"You must pass a positive integer to the method {metd} !")

    def _parse_param_list(self, args):
        other_filter = [i for i in args if isinstance(i, AdvancedNodeSet)]
        queries = [i for i in args if isinstance(i, Q) or isinstance(i, django.db.models.Q)]
        return other_filter, queries

    def __len__(self):
        return len(self.fetch_all())

    def __getitem__(self, item):
        if not self._cache:
            if item == 0:
                return self.first()
            self.fetch_all()
        return self._cache[item]

    def __iter__(self):
        return self.Iterator(self.fetch_all())

    def __bool__(self):
        return self.first() is not None

    def __or__(self, other):
        if isinstance(other, AdvancedNodeSet):
            if other.cls != self.cls:
                self._filter |= Q(_=other)
            else:
                self._filter |= other._filter
        if isinstance(other, Q):
            self._filter |= other
        return self

    def __and__(self, other):
        if isinstance(other, AdvancedNodeSet):
            if other.cls != self.cls:
                self._filter &= Q(_=other)
            else:
                self._filter &= other._filter
        if isinstance(other, Q):
            self._filter &= other
        return self

    def __hash__(self):
        return hash(self.__class__)

    class Iterator:
        def __init__(self, data):
            self.pointer = 0
            self.data = data

        def __next__(self):
            if self.pointer < len(self.data):
                ret = self.data[self.pointer]
                self.pointer += 1
                return ret
            else:
                raise StopIteration
