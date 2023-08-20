# neomodel_plus
An enhanced version of Neomodel (django_neomodel)


# Implemented a query builder similar to Django-ORM API to simplify the use of Neomodel

### You can find use examples in `demo.py`

```python
# this code can retrieval images taken on 2020 by the user whose name is 'admin'
Image.filter(user__username="admin", date=Date.filter(year=2020))

# You can use operators `&` or `|` to combine multiple filters
Image.filter(tag=Tag.filter(name__startswith='d')) & Image.filter(size__gte=30)

# You can combine filter with Q object
Image.filter(user__username='root') & Q(tag=Tag.filter(name='a')


# You can use the prefix `rel_` to filter images based on the property `max_score` which is defined on the relationship between `Image` and `Tag`
Image.filter(Q(rel_tag__max_score__gte=10))


# `skip`, `limit` and `order by` are also available
Image.filter(...).skip(10).limit(10).orderby('size')
```


# Enhanced all property classes

###  by supporting relationship property indexing
```python
class HasTag(StructuredRel):
    rel = 'has_tag'
    max_score = FloatProperty(index=True)
```

###  by supporting `full text`  index
```python
class Book(RelSearchableNode):
    content = StringProperty(full_text_index=True)
```

# New base class
All Node classes are now  subclasses of `RelSearchableNode`, it provides common static methods like `get`, `filter`, `get_or_create`, etc.
```python
class RelSearchableNode(DjangoNode):
    ...
    
    class Meta:
        # Default selected properties for each search
        # e.g: default_select = ('name', 'age')
        # For each search returns only selected properties, in this case they are `name` and `age`
        # If `default_select` is not declared or is empty, then all properties of the current Node will be returned
        default_select = ()
    
        # equivalent to `unique_together` of Django ORM
        # you can use it to create composite indexes
        # e.g. to create an index composited by 2 properties `name` and `age` you can do: composite_index.append(('name', 'age')) 
        # e.g. if you want to create a composite full text index you should insert `1` at the position zero: composite_index.append((1, 'name', 'age')) 
        composite_index = []
```

# How to use
For now, it is necessary to download this repository, and put it into your project.

# How to create index
You can find following functions in `utils.py`
+ `install_neo_indexes`, delete and reinstall all user indexes (created by user)
+ `drop_indexes`, delete all user indexes
+ `create_all_indexes`, create all scanned indexes
+ `clean_db`, remove all data


# Used libraries
* Neomodel: https://github.com/neo4j-contrib/neomodel
* Django_neomodel: https://github.com/neo4j-contrib/django-neomodel

# TODO
Complete search functionality with full text.


