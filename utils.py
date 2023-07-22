import json
import importlib
import os
import re
from collections import defaultdict
from collections.abc import Iterable

from django.conf import settings

from neomodel import db, StructuredRel, config, StructuredNode
from neomodel.contrib.spatial_properties import PointProperty
from neomodel.properties import Property

try:
    DATABASE_URL = settings.NEOMODEL_NEO4J_BOLT_URL
except:
    DATABASE_URL = 'bolt://neo4j:root@localhost:7687'

config.DATABASE_URL = DATABASE_URL

def is_collection(v):
    return isinstance(v, Iterable) and not isinstance(v, (str, bytes, bytearray))


def class_for_name(file_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(file_name + '.models')
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def subsub(cls):  # recursively return all subclasses
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in subsub(s)]


def install_neo_indexes():
    print("Dropping all indexes and constraints...")
    drop_indexes()
    print("All indexes and constrains dropped")
    print()
    print("Creating indexes...")
    create_all_indexes()
    print("All indexes were created")


def create_all_indexes():
    load_node_classes()
    for cls in subsub(StructuredNode):
        create_index(cls, True)

    for cls in subsub(StructuredRel):
        create_index(cls, False)


def create_index(cls, node):
    cls_properties = cls.defined_properties(aliases=False, rels=False).items()
    for name, property in cls_properties:
        if node:
            # Node
            if isinstance(property, Property):
                if property.index or (hasattr(property, 'need_index') and property.need_index):
                    if isinstance(property, PointProperty):
                        db.cypher_query(f"CREATE POINT INDEX FOR (n:{cls.__label__}) ON (n.{name})")
                    else:
                        db.cypher_query(f"CREATE INDEX FOR (n:{cls.__label__}) ON (n.{name})")
                    print(f"Created index for Node ['{cls.__label__}'] on property ['{name}']")
                elif hasattr(property, 'full_text_index') and property.full_text_index:
                    index_name = f'full_text_{cls.__label__}_{name}'
                    db.cypher_query(f"CREATE FULLTEXT INDEX {index_name} FOR (n:{cls.__label__}) ON EACH [n.{name}]")
                    cls.full_text_index[name] = index_name
                    print(f"Created index for Node ['{cls.__label__}'] on property ['{name}']")
                elif property.unique_index:
                    db.cypher_query(f"CREATE CONSTRAINT FOR (n:{cls.__label__}) REQUIRE n.{name} IS UNIQUE")
                    print(f"Created unique constraint for ['{name}'] on Node ['{cls.__label__}']")
        else:
            # Relationship
            if isinstance(property, Property):
                if property.index or (hasattr(property, 'need_index') and property.need_index):
                    # for relationship property indexing
                    db.cypher_query(f"CREATE INDEX FOR ()-[n:{cls.rel}]-() ON (n.{name})")
                    print(f"Created index for Relationship ['{cls.rel}'] on property ['{name}'] ")
                elif property.unique_index:
                    print(f"Unique constraint for a Relationship property is not supported right now")

    if hasattr(cls, 'Meta') and hasattr(cls.Meta, 'composite_index'):
        for composite in cls.Meta.composite_index:
            full_text = composite[0] == 1
            if full_text:
                composite = composite[1:]
            on_prop = ''
            for prop_name in composite:
                on_prop += f'n.{prop_name},'
            on_prop = on_prop[:-1]
            if issubclass(cls, StructuredNode):
                if full_text:
                    index_name = f'full_text_{cls.__label__}_{"_".join(sorted(composite))}'
                    db.cypher_query(f"CREATE FULLTEXT INDEX {index_name} FOR (n:{cls.__label__}) ON EACH [{on_prop}]")
                    print(f"Created composite full-text index for Node ['{cls.__name__}'] on properties {composite}")
                else:
                    db.cypher_query(f"CREATE INDEX FOR (n:{cls.__label__}) ON ({on_prop})")
                    print(f"Created composite index for Node ['{cls.__name__}'] on properties {composite}")
            elif issubclass(cls, StructuredRel):
                db.cypher_query(f"CREATE INDEX FOR ()-[n:{cls.rel}]-() ON ({on_prop})")
                print(f"Created composite index for Relationship ['{cls.__name__}'] on properties {composite}")


def drop_indexes():
    """
    Discover and drop all user indexes.
    """

    results, meta = db.cypher_query("SHOW INDEX")
    for index in results:
        try:
            if index[0] in (1, 2):
                # system indexes
                continue

            index_type = 'CONSTRAINT'
            if index[1].startswith("index"):
                index_type = 'INDEX'

            if index[4] == 'UNIQUE':
                # drop constraints
                db.cypher_query(f'DROP {index_type} {index[1]}')
                print(f"Dropping UNIQUE index for Node {index[6]} on property {index[7]}")
            elif index[4] in ("RANGE", "BTREE", "POINT"):
                # drop btree index
                db.cypher_query(f'DROP {index_type} {index[1]}')
                print(f"Dropped RANGE index for Node {index[6]} on property {index[7]}")
            elif index[4] == 'FULLTEXT':
                db.cypher_query('DROP INDEX ' + index[1])
                print(f"Dropped FULLTEXT index for Node {index[6]} on property {index[7]}")
        except Exception as e:
            print(e)


def clean_db():
    config.DATABASE_URL = DATABASE_URL
    ops = input("Do you really want to delete all nodes and relationships? [y/n] ")

    if ops.lower() in ('y', 'yes'):
        print("Deleting all nodes and relationships...")
        node_count, _ = db.cypher_query("MATCH (a) RETURN COUNT(a)")
        rel_count, _ = db.cypher_query("MATCH p=()-[r]->() RETURN COUNT(r) ")
        node_count = node_count[0][0]
        rel_count = rel_count[0][0]
        for _ in range(rel_count // 10000 + 1):
            db.cypher_query("MATCH p=()-[r]->() WITH r LIMIT 10000 DELETE r")
        print("All relationships were deleted")

        for _ in range(node_count // 10000 + 1):
            db.cypher_query("MATCH (p) WITH p LIMIT 10000 DELETE p")

        db.cypher_query("CALL spatial.deletePolygons()")
        print("All nodes were deleted")
    else:
        print("Operation canceled")

def load_data(args):
    config.DATABASE_URL = DATABASE_URL
    counter = defaultdict(int)
    for arg in args:
        if os.path.isdir(arg):
            for path, _, files in os.walk(arg):
                for file in files:
                    file_path = os.path.join(path, file)
                    load_data_from_json(file_path, counter)
        else:
            load_data_from_json(arg, counter)

    for k, v in counter.items():
        print(f"Created {v} [{k}] nodes in database")

def load_data_from_json(file, counter):
    with open(file, 'rb') as f:
        neo_objs = json.load(f)
        for neo_obj in neo_objs:
            model = neo_obj['model']
            last_point = model.rindex('.')
            file_name, class_name = model[:last_point], model[last_point + 1:]
            clazz = class_for_name(file_name, class_name)
            clazz.get_or_create(**neo_obj['fields'])
            counter[class_name] += 1



def merge_dict(source, target):
    for k in source:
        target[k].extend(source[k])


def name_convert_to_snake(name: str) -> str:
    if '_' not in name:
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    return name.lower()


allNodes = None


def str_to_class(class_name: str):
    """
    get class object from class name
    """
    global allNodes
    if not allNodes:
        load_node_classes()
        allNodes = {c.__name__.lower(): c for c in subsub(StructuredNode)}
    return allNodes[class_name.lower()]


def load_node_classes():
    import pkgutil, importlib, os
    try:
        pkg_dir = str(settings.BASE_DIR)
    except:
        pkg_dir = os.path.dirname(__file__)  # neomodel_plus
        pkg_dir = os.path.dirname(pkg_dir)  # MEMORIA
    for module_loader, name, ispkg in pkgutil.iter_modules([pkg_dir]):
        try:
            importlib.import_module(name + ".models", __package__)
        except:
            pass
    global allNodes
    allNodes = {c.__name__.lower(): c for c in subsub(StructuredNode)}


def load_full_text_index(cls):
    results, _ = db.cypher_query("SHOW INDEX")
    for r in results:
        if r[4] == 'FULLTEXT' and r[6][0] == cls.__label__:
            cls.full_text_index = r[1]
