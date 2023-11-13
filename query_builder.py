import datetime
import inspect
from collections import defaultdict
from io import StringIO

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
    def __init__(self, rel_path: "RelPath", keyword_argument: KeywordArgument):
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
        self.left = self.right = '-'
        to = rel_direction == 1
        if to:
            self.right = '->'
        else:
            self.left = '<-'
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


OPERATORS = _OPERATORS_CLASS()


class CypherKeyWords:
    class KeyWord:
        def __init__(self, name: str, alias=None):
            self.name = name
            self.alias = alias

        def __eq__(self, other):
            if isinstance(other, str):
                return other == self.name or other in self.alias

            elif isinstance(other, CypherKeyWords.KeyWord):
                return id(other) == id(self)

            return False

        def __repr__(self):
            return self.name

        def __add__(self, other):
            return self.name + other

    RETURN = KeyWord(name="RETURN")
    MATCH = KeyWord(name="MATCH")
    WHERE = KeyWord(name="WHERE")
    WITH = KeyWord(name="WITH")
    AND = KeyWord(name="AND", alias={"and"})
    NOT = KeyWord(name="NOT")
    OR = KeyWord(name="OR", alias={"or"})
    UNION = KeyWord(name="UNION")
    DISTINCT = KeyWord(name="DISTINCT")
    ORDER_BY = KeyWord(name="ORDER BY")
    SKIP = KeyWord(name="SKIP")
    LIMIT = KeyWord(name="LIMIT")


class Variable:
    def __init__(self, newly_created, name, flush=False):
        self.newly_created = newly_created
        self.name = name
        self.flush = flush

    def __str__(self):
        return f'newly_created: {self.newly_created}, name: {self.name}, flush: {self.flush}'

    def __repr__(self):
        return str(self)


class ASTTree:
    def __init__(self, root: "Statement" = None):
        self.root: "Statement" = root if root else Statement(None)
        self.leaf: "Statement" = self.root

    def add_to_last(self, stmt: "Statement"):
        self.leaf.child = stmt
        self.leaf = stmt

    def add_to_head(self, stmt: "Statement"):
        temp = self.root.child
        stmt.child = temp
        self.root.child = stmt

    def get_cypher(self):
        return self.root.get_cypher().cypher

    def get_var(self, key):
        return self.root._get_var(key)

    def get_type_counter(self, type):
        return self.root._get_type_counter(type)


class Statement:
    def __init__(self, node, counter=0):
        self.variable_name = None
        self.parent = None
        self.child = None
        self._node = node
        self._counter = counter
        self._variable_table = {}
        self._type_counter = defaultdict(int)

    def get_cypher(self, parent: "Statement" = None) -> "StatementResult":
        parent = parent if parent else self
        if self.child:
            return self.child.get_cypher(parent)
        return StatementResult("", None)

    def _next_var(self, key):
        if self.parent:
            return self.parent._next_var(key)
        else:
            """
            create a variable for the given key
            """
            variable = self._variable_table.get(key)
            if not variable:
                variable = Variable(True, f"var_{key}_{self._get_type_counter(key)}")
                self._variable_table[key] = variable
            else:
                if variable.flush:
                    self._update_type_counter(key)
                    self._variable_table[key] = None
                variable.newly_created = False
            return variable

    def _get_var(self, key):
        if self.parent:
            return self.parent._get_var(key)
        else:
            return self._variable_table.get(key)

    def _clear_variable_table(self, clear_all=True):
        if self.parent:
            self.parent._clear_variable_table()
        else:
            if clear_all:
                self._variable_table.clear()
            else:
                self._variable_table = {
                    self._node: self._variable_table[self._node]}

    def _get_type_counter(self, type):
        if self.parent:
            return self.parent._get_type_counter(type)
        else:
            c = self._type_counter[type]
            if c == 0:
                c = self._counter
            self._type_counter[type] = c
            return c

    def _update_type_counter(self, type, newV=None):
        if self.parent:
            return self.parent._update_type_counter(type, newV)
        else:
            if newV is not None:
                self._type_counter[type] = max(newV, self._type_counter[type])
            else:
                c = self._get_type_counter(type)
                c += 1
                self._type_counter[type] = c
                return c

    def _update_variable_table(self, node_type, new_var):
        if self.parent:
            self.parent._update_variable_table(node_type, new_var)
        else:
            self._variable_table[node_type] = new_var

    def _need_reflush(self, node_type):
        if self.parent:
            self.parent._need_reflush(node_type)
        else:
            variable = self._variable_table[node_type]
            self._variable_table[node_type] = Variable(False, variable.name, flush=True)


class Query(Statement):
    def __init__(self, node, parent: "Statement" = None):
        super().__init__(node)
        if parent:
            self.return_var = parent._get_var(node)


class RecursiveStatement(Statement):
    def __init__(self, rel_kw: "RelKeywordArgument", parent_filter_cls=None):
        super().__init__(None)
        self.rel_kw = rel_kw
        self.parent_filter_cls = parent_filter_cls

    def get_cypher(self, parent: "Statement" = None):
        self.parent = parent
        adv_filter = self.rel_kw.keyword_argument.value
        node_name = adv_filter.cls.__name__
        c = self._update_type_counter(type=node_name)
        result = QueryBuilder(adv_filter)._build_query(without_return=True, counter=c)
        self._update_type_counter(type=node_name, newV=result.type_counter + 1)
        new_node_name = f'{node_name}_{result.type_counter}'
        self._update_variable_table(new_node_name, Variable(False, result.returned_var))
        self.rel_kw.rel_path.node_name = new_node_name

        if self.child:
            child_result = self.child.get_cypher(parent)
            return StatementResult(f'{result.cypher}{child_result.cypher}', child_result.returned_var)
        filter_result = QueryBuilder(adv_filter)._build_query(without_return=True)
        return StatementResult(f'{filter_result.cypher}', self._get_var(node_name))


class MergeStatement(Statement):
    def __init__(self, node, connector: CypherKeyWords.KeyWord):
        super().__init__(node)
        self.connector = connector

    def get_cypher(self, parent: "Statement" = None):
        self.parent = parent
        if self.child:
            return StatementResult(f'{self._get_cypher().cypher}{self.child.get_cypher(parent).cypher}', None)
        return StatementResult(self._get_cypher().cypher, None)

    def _get_cypher(self):
        cypher = ''
        # if there are more queries, add the keyword UNION if the connector is OR
        if self.connector == CypherKeyWords.OR:
            cypher = f'{CypherKeyWords.RETURN} {self._next_var(self._node).name}\n{CypherKeyWords.UNION}\n'
            self._clear_variable_table()
        elif self.connector == CypherKeyWords.AND:
            cypher = f'{CypherKeyWords.WITH} {self._next_var(self._node).name}\n'
            self._clear_variable_table(False)
        return StatementResult(cypher, None)


class MatchStatement(Statement):
    def __init__(self, node: str):
        super().__init__(node)
        self.variable_name = None
        self.child = None

    def get_cypher(self, parent: "Statement" = None) -> "StatementResult":
        self.parent = parent
        if self.child:
            result = self._get_cypher()
            child_result = self.child.get_cypher(parent)
            return StatementResult(result.cypher + ' ' + child_result.cypher, result.returned_var)
        else:
            return self._get_cypher()

    def _get_cypher(self):
        variable = self._next_var(self._node)
        self.variable_name = variable.name
        if variable.newly_created:
            stmt = f'{CypherKeyWords.MATCH} ({variable.name}:{self._node})'
        else:
            stmt = f'{CypherKeyWords.MATCH} ({variable.name})'
        return StatementResult(stmt, variable.name)


class RelationStatement(Statement):
    def __init__(self, start_node, relationship_def: RelKeywordArgument):
        super().__init__(start_node)
        self.relationship_def = relationship_def
        kw = relationship_def.keyword_argument
        self.match_stmt = MatchStatement(start_node)
        self.search_type = kw.type
        if kw.type == SearchType.REL_PROP:
            self.where_stmt = WhereStatement(relationship_def.rel_path.rel_type,
                                             AttributeConditions([relationship_def]))
        else:
            self.where_stmt = WhereStatement(relationship_def.rel_path.rel_type, AttributeConditions())

    def get_cypher(self, parent: "Statement" = None):
        self.parent = parent
        result = self._get_cypher()
        if self.search_type != SearchType.REL_PROP:
            where_cypher = "\n"
        elif self.search_type == SearchType.REL_PROP:
            where_cypher = self.where_stmt.get_cypher(parent).cypher
        else:
            where_cypher = ""

        if self.child:
            child_result = self.child.get_cypher(parent)
            return StatementResult(f'{result.cypher} {where_cypher}{child_result.cypher}', result.returned_var)
        return StatementResult(result.cypher + ' ' + where_cypher, result.returned_var)

    def _get_cypher(self):
        rel_path = self.relationship_def.rel_path
        rel_node_name = rel_path.node_name
        rel_variable = self._next_var(rel_path.rel_type)

        connected_node_variable = self._next_var(rel_node_name)
        if connected_node_variable.newly_created:
            # if this variable has already been created
            rel_stmt = self._write_relationship_stmt(rel_path, rel_variable.name,
                                                     connected_node_variable.name, declare=True)
        else:
            rel_stmt = self._write_relationship_stmt(rel_path, rel_variable.name,
                                                     connected_node_variable.name)

        match_result = self.match_stmt.get_cypher(self.parent)
        return StatementResult(match_result.cypher + rel_stmt, match_result.returned_var)

    def _write_relationship_stmt(self, rel_def: RelPath, rel_variable_name, connected_node_variable_name,
                                 declare=False):
        if declare:
            return f"{rel_def.left}[{rel_variable_name}:`{rel_def.rel_type}`]{rel_def.right}({connected_node_variable_name}:{rel_def.node_name})"
        else:
            return f"{rel_def.left}[{rel_variable_name}:`{rel_def.rel_type}`]{rel_def.right}({connected_node_variable_name})"


class WhereStatement(Statement):
    def __init__(self, node, where_conditions: "AttributeConditions"):
        super().__init__(node)
        self.node = node
        self.where_conditions = where_conditions
        self.cypher = StringIO()

    def get_cypher(self, parent: "Statement" = None):
        self.parent = parent
        variable = self._next_var(self.node)
        if self.where_conditions.degree() > 0:
            self._write(CypherKeyWords.WHERE.name, space=True)

        self._write_where_stmt(StructuredNode, self.where_conditions.and_conditions, CypherKeyWords.AND,
                               variable.name)

        if self.where_conditions.degree() > 1:
            self._add_merge_key_word(CypherKeyWords.AND)

        self._write_where_stmt(StructuredNode, self.where_conditions.or_conditions, CypherKeyWords.OR,
                               variable.name)

        cypher = self.cypher.getvalue()
        if cypher:
            cypher += '\n'

        if self.child:
            child_result = self.child.get_cypher(parent)
            return StatementResult(cypher + child_result.cypher, child_result.returned_var)
        else:
            return StatementResult(cypher, None)

    def _write_where_stmt(self, target_type: type, condition_list: list, connector: CypherKeyWords.KeyWord, var: str,
                          continuation: bool = False):
        """
        Write where statement for the current MATCH query
        """
        if not condition_list:
            return

        if connector == CypherKeyWords.OR and len(condition_list) > 1:
            self._write('(')

        self._write_property_filters(condition_list, connector, target_type, var)

        if connector == CypherKeyWords.OR and len(condition_list) > 1:
            self._write(') ')

    def _write_property_filters(self, condition_list: list, connector: CypherKeyWords.KeyWord, target_type: type,
                                var: str):
        for j, condition in enumerate(condition_list):
            if isinstance(condition, RelKeywordArgument):
                rel_path, condition = condition

            negated, param, operator, v, type = condition
            not_ = CypherKeyWords.NOT + ' ' if negated else ''

            if type == SearchType.ID:
                if is_collection(v):
                    self._write_id_stmt(not_, var, operator, _convert_nodes_to_ids(target_type, v))
                else:
                    _check_type(v, target_type)
                    v = _to_id(v)
                    self._write_id_stmt(not_, var, OPERATORS.EQ, v)

            elif type == SearchType.REL_PROP:
                rel_type = rel_path.rel_type
                self._write_value_stmt(not_, self._next_var(rel_type).name, param, operator,
                                       _change_if_v_not_adequate(v))
            else:
                if operator == OPERATORS.IN and not isinstance(v, list):
                    v = list(v) if isinstance(v, tuple) else [v]

                self._write_value_stmt(not_, var, param, operator, _change_if_v_not_adequate(v))

            if j + 1 < len(condition_list):
                self._add_merge_key_word(connector)

    def _add_merge_key_word(self, connector: CypherKeyWords.KeyWord):
        self._write(connector.name, space=True)

    def _write(self, s, space=False):
        if space:
            self.cypher.write(s + ' ')
        else:
            self.cypher.write(s)

    def _new_line(self):
        self._write("\n")

    def _write_id_stmt(self, negated: str, variable_name: str, operator: str, value: int):
        self._write(f"{negated}ID({variable_name}) {operator} {value} ")

    def _write_value_stmt(self, negated: str, variable_name: str, param: str, operator: str, value):
        self._write(f"{negated}{variable_name}.{param} {operator} {value} ")


class ReturnStatement(Statement):
    def __init__(self, node_detail: NodeRelationshipDetail, filter: "AdvancedNodeSet", without_return=False):
        super().__init__(node_detail.cls_name)
        self.filter = filter
        self.node_detail = node_detail
        self.without_return = without_return
        self.cypher = StringIO()

    def get_cypher(self, parent: "Statement" = None):
        self.parent = parent
        child_result = self.child.get_cypher(parent)
        result = self._get_cypher(child_result.returned_var)
        return StatementResult(child_result.cypher + result.cypher, child_result.returned_var)

    def _get_cypher(self, variable=None):
        aggregate = self.filter._aggregate
        if not self.without_return:
            if variable is None:
                variable = self._next_var(self._node).name
            count_records = self.filter._count_records
            orderby_params = self.filter._order_by

            if not aggregate and not count_records:
                orderby_params = self._create_orderby_params(variable, orderby_params)

            self._write_return_stmt(variable, aggregate, count_records)

            if not aggregate:
                self._write_order_by_stmt(orderby_params)

                self._write_skip_stmt()

                self._write_limit_stmt()
            self._write("\n")
        return StatementResult(self.cypher.getvalue(), variable)

    def _write(self, s, space=True):
        if space:
            self.cypher.write(s + ' ')
        else:
            self.cypher.write(s)

    def _write_limit_stmt(self):
        lmt = self.filter._limit
        if lmt:
            self._write(f"{CypherKeyWords.LIMIT} {lmt} ")

    def _write_skip_stmt(self):
        skp = self.filter._skip
        if skp:
            self._write(f"{CypherKeyWords.SKIP} {skp} ")

    def _write_order_by_stmt(self, orderby_params):
        if orderby_params:
            orderby_stmt = ', '.join([f"{o[0]}.{o[1]} {o[2]}" for o in orderby_params])
            self._write(f" {CypherKeyWords.ORDER_BY} {orderby_stmt}")

    def _write_return_stmt(self, variable: str, aggregate: Aggregate, count=False):
        if aggregate:
            self._write(f' {CypherKeyWords.RETURN} {aggregate.name}({variable}.{aggregate.identity[1][1][0]})')
            return

        self._write_distinct_stmt(variable)

        self._write(f' {CypherKeyWords.RETURN.name} ')

        if count:
            self._write(f' COUNT({variable})')
        else:
            select = self.filter._select
            if select:
                self._transform_select(select)
                self._write_select_stmt(select, variable)
            else:
                self._write(variable)

    def _transform_select(self, select):
        for i, prop in enumerate(select):
            if not self._check_if_current_node_property(prop):
                if prop == 'id':
                    select[i] = (True, prop)
                else:
                    raise AttributeError(
                        f"{self.node_detail.cls_name} does not have property '{prop}'!")
            else:
                select[i] = (False, prop)

    def _write_select_stmt(self, select: list, variable: str):
        for i, prop in enumerate(select):
            id_prop, prop = prop
            if i > 0:
                self._write(", ")

            if id_prop:
                self._write(f"ID({variable}) ")
            else:
                self._write(f"{variable}.{prop} ")

    def _write_distinct_stmt(self, variable: str):
        if self.filter._distinct:
            self._write(f"{CypherKeyWords.WITH} {CypherKeyWords.DISTINCT} {variable} ")

    def _create_orderby_params(self, cls_var, odby):
        orderby_params = []

        # verify if order_by properties belong to the current node or to some node directly connected with it
        for o in odby:
            prop, order = o
            is_current_node_property = self._check_if_current_node_property(prop)

            # check if is a property of a directly connected node
            is_directly_connected_node_property = self._check_if_connected_node_property(self.node_detail.cls, prop)

            if not is_current_node_property and not is_directly_connected_node_property and prop != 'id':
                raise AttributeError(
                    f"{self.node_detail.cls_name} does not have property '{prop}'!")

            orderby_params.append((cls_var, prop, order))

        return orderby_params

    def _check_if_current_node_property(self, prop_name):
        return self.node_detail.is_node_property(prop_name)

    def _check_if_connected_node_property(self, clzz, prop_name):
        self.node_detail.get_linkedNode_cls(clzz.__name__)
        directly_connected_node = _get_node_detail(clzz)
        return directly_connected_node.is_node_property(prop_name)


class Conditions:
    def __init__(self, and_conditions, or_conditions):
        self.and_conditions = and_conditions
        self.or_conditions = or_conditions

    def __bool__(self):
        return len(self.and_conditions) + len(self.or_conditions) > 0

    def degree(self):
        return int(bool(self.and_conditions)) + int(bool(self.or_conditions))


class RelConditions(Conditions):
    def __init__(self):
        super().__init__(defaultdict(list), defaultdict(list))

    def __bool__(self):
        return len(self.and_conditions) + len(self.or_conditions) > 0

    def degree(self):
        return int(bool(self.and_conditions)) + int(bool(self.or_conditions))

    def add(self, key, value, AND=True):
        if AND:
            self.and_conditions[key].append(value)
        else:
            self.or_conditions[key].append(value)

    def extend(self, key, list_, AND=True):
        if AND:
            self.and_conditions[key].extend(list_)
        else:
            self.or_conditions[key].extend(list_)

    def merge(self, rel_conditions: "RelConditions"):
        def _merge(dictA, dictB):
            for k in dictB.keys():
                dictA[k].extend(dictB[k])

        _merge(self.and_conditions, rel_conditions.and_conditions)
        _merge(self.or_conditions, rel_conditions.or_conditions)


class AttributeConditions(Conditions):
    def __init__(self, and_conditions=None, or_conditions=None):
        super().__init__(and_conditions, or_conditions)
        if or_conditions is None:
            self.or_conditions = []
        if and_conditions is None:
            self.and_conditions = []

    def __getitem__(self, item):
        if item == CypherKeyWords.AND:
            return self.and_conditions
        return self.or_conditions

    def add(self, value, AND=True):
        if AND:
            self.and_conditions.append(value)
        else:
            self.or_conditions.append(value)

    def merge(self, attr_conditions: "AttributeConditions"):
        self.and_conditions.extend(attr_conditions.and_conditions)
        self.or_conditions.extend(attr_conditions.or_conditions)


class AttributeRelConditions:

    def __init__(self):
        # attribute conditions for the current Node
        self.rel_conditions = RelConditions()

        # relationships directly used by the current Node
        self.attr_conditions = AttributeConditions()

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
        param_list = remove_duplication(param_list)
        is_AND = connector == CypherKeyWords.AND
        conditions = AttributeRelConditions()

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

                rel_kwarg = self._convert_rel_condition_to_argument(split_param[0][4:],
                                                                    KeywordArgument(negated, split_param[1], operator,
                                                                                    v,
                                                                                    type=SearchType.REL_PROP))
                conditions.add_rel_condition(split_param[0][4:], rel_kwarg, is_AND)
            else:
                # relationship search
                if len(split_param) == 1:
                    if isinstance(v, AdvancedNodeSet):
                        # e.g: Image.filter(tag=Tag.filter(name='a')) where `tag` is a declared property in Image
                        # class that represents the relationship between Image and Tag
                        rel_kwarg = self._convert_rel_condition_to_argument(split_param[0],
                                                                            KeywordArgument(negated, None, operator, v,
                                                                                            type=SearchType.RECUR_NSET))

                    else:
                        # e.g: Image.filter(tag=Tag())
                        rel_kwarg = self._convert_rel_condition_to_argument(split_param[0],
                                                                            KeywordArgument(negated, None, operator, v,
                                                                                            type=SearchType.ID))
                else:
                    # e.g: Image.filter(tag__name__startswith='a')
                    rel_kwarg = self._convert_rel_condition_to_argument(split_param[0],
                                                                        KeywordArgument(negated,
                                                                                        f"{'__'.join(split_param[1:])}",
                                                                                        operator, v,
                                                                                        type=SearchType.RECUR_PROP))
                conditions.add_rel_condition(split_param[0], rel_kwarg, is_AND)
        return conditions

    def _convert_rel_condition_to_argument(self, connected_node_name, kwarg: KeywordArgument):
        rel_kwarg = None
        param, v, type = kwarg.parameter, kwarg.value, kwarg.type
        if type != SearchType.ID and not self.node_cls_detail.is_node_relationship(connected_node_name):
            if type == SearchType.RECUR_NSET:
                # There is no defined relationship between Image and Tag in the class Image
                # But we have the relationship declared in the class Tag
                # So we can do something like: Image.filter(tag=Tag.filter(name='abc'))
                detail = _get_node_detail(v.cls)
                if detail.is_connected_with(self.node_cls_detail.cls):
                    rel_path = detail.get_rel_detail(self.node_cls_detail.cls_name, inverse=True)
                    rel_kwarg = RelKeywordArgument(rel_path, kwarg)
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
                    connected_class = str_to_class(connected_node_name)
                    detail = _get_node_detail(connected_class)
                    if detail.is_connected_with(self.node_cls_detail.cls):
                        rel_path = detail.get_rel_detail(self.node_cls_detail.cls_name, inverse=True)
                        if type in (SearchType.ID, SearchType.REL_PROP):
                            # e.g: Image.filter(tag=Tag())
                            rel_kwarg = RelKeywordArgument(rel_path, kwarg)
                        else:
                            # type == SearchType.RECUR_PROP
                            # e.g: Image.filter(tag__name='a')
                            rel_kwarg = RelKeywordArgument(rel_path,
                                                           KeywordArgument(kwarg.negated,
                                                                           kwarg.parameter,
                                                                           kwarg.operator,
                                                                           detail.cls.filter(
                                                                               **{param: v})))
                    else:
                        self._no_property_error(connected_node_name)
                except KeyError:
                    self._no_property_error(connected_node_name)

        else:
            rel_path = self.node_cls_detail.get_rel_detail(connected_node_name)
            if type == SearchType.RECUR_PROP:
                # need to be recursively processed
                rel_kwarg = RelKeywordArgument(rel_path,
                                               KeywordArgument(kwarg.negated, kwarg.parameter,
                                                               kwarg.operator,
                                                               rel_path.cls.filter(**{param: v})))
            else:
                # type == SearchType.RECUR_NSET
                rel_kwarg = RelKeywordArgument(rel_path, kwarg)

        return rel_kwarg

    def _no_property_error(self, rel):
        raise ValueError(
            f"There is no property or connected Node called {rel} in {self.node_cls_detail.cls_name}")


class StatementBuilder:
    def convert_params_to_statements(self, ast: ASTTree, node_name, conditions: AttributeRelConditions):

        if conditions.attr_conditions.degree() > 0:
            statement = MatchStatement(node_name)
            ast.add_to_last(statement)
            where_stmt = WhereStatement(node_name, conditions.attr_conditions)
            ast.add_to_last(where_stmt)

        def _rel_stmt(conditions, connector):
            for key in conditions:
                for i, value in enumerate(conditions[key]):
                    value: RelKeywordArgument = value
                    if isinstance(value.keyword_argument.value, AdvancedNodeSet):
                        stmt = RecursiveStatement(value)
                        ast.add_to_last(stmt)
                    stmt = RelationStatement(node_name, value)
                    ast.add_to_last(stmt)

                    if i + 1 < len(conditions[key]):
                        merge = MergeStatement(node_name, connector)
                        ast.add_to_last(merge)

        _rel_stmt(conditions.rel_conditions.and_conditions, CypherKeyWords.AND)
        _rel_stmt(conditions.rel_conditions.or_conditions, CypherKeyWords.OR)

        if conditions.attr_conditions.degree() == 0 and not isinstance(ast.leaf, RelationStatement):
            # if there is no where statement for the current node, we can place the match statement at the end
            ast.add_to_last(MatchStatement(node_name))


class QueryBuilderResult:
    def __init__(self, cypher, returned_var, type_counter):
        self.cypher = cypher
        self.returned_var = returned_var
        self.type_counter = type_counter


class StatementResult:
    def __init__(self, cypher, returned_var):
        self.cypher = cypher
        self.returned_var = returned_var


def _space(space):
    return " " if space else ""


def _convert_nodes_to_ids(target_type, nodes):
    """
    convert a list of nodes to a list of ids
    """
    check = _check_type

    if is_collection(nodes):
        if nodes:
            return [_to_id(e) for e in nodes if check(e, target_type)]
        raise ValueError("Empty collection detected")
    return nodes


def _to_id(e):
    if isinstance(e, int):
        return e
    else:
        return e.id


def _check_type(given_obj, target_type):
    """
    check type of given object
    """
    if not isinstance(given_obj, StructuredNode) and \
            not isinstance(given_obj, target_type) and \
            not isinstance(given_obj, int):
        raise ValueError(f"Invalid ID values!")

    return True


def _change_if_v_not_adequate(v):
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, datetime.datetime):
        return v.timestamp()
    elif isinstance(v, datetime.date):
        return f"'{str(v)}'"
    return v


def remove_duplication(params):
    r = []
    for i in params:
        if i not in r:
            r.append(i)
    return r


NODE_DETAILS = {}


def _get_node_detail(clazz: type):
    details = NODE_DETAILS.get(clazz.__name__)
    if details is None:
        details = NodeRelationshipDetail(clazz)
        NODE_DETAILS[clazz.__name__] = details
    return details


class QueryBuilder:
    def __init__(self, advanced_filter):
        self.advanced_filter = advanced_filter
        self.node_cls_detail = _get_node_detail(advanced_filter.cls)
        self.parser = QueryParser(self.node_cls_detail)
        self.statement_builder = StatementBuilder()

    def _build_query(self, without_return=False, counter=0):
        self.ast = ASTTree(Statement(self.node_cls_detail.cls_name, counter=counter))
        queries = self.advanced_filter._filter
        cls_name = self.node_cls_detail.cls_name
        if len(queries.children) == 0:
            # when user passes an empty Q, e.g: Image.filter(Q())
            self.ast.add_to_last(MatchStatement(cls_name))
        else:

            # e.g: query_1 = Q(age=1), query_2 = Q(name_startswith='d')
            # c = query_1 & query_2. In this case, the connector between the queries is 'AND'
            # c = query_1 | query_2. The connector between the queries is 'OR'
            # if convert these queries to cypher, they will be a combination of MATCH queries
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
                    self.ast.add_to_last(MatchStatement(cls_name))
                else:
                    conditions = self.parser.parse_param_list(inter_param_connector, param_list,
                                                              negated)
                    self.statement_builder.convert_params_to_statements(self.ast,
                                                                        cls_name,
                                                                        conditions)
        return_stmt = ReturnStatement(self.node_cls_detail, self.advanced_filter, without_return)
        self.ast.add_to_head(return_stmt)

        result = QueryBuilderResult(self.ast.get_cypher(), self.ast.get_var(cls_name).name,
                                    self.ast.get_type_counter(cls_name))
        return result


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
