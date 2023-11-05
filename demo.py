from neomodel import Q, StructuredRel, One
from neomodel.relationship_manager import RelationshipTo, RelationshipFrom
from properties import StringProperty, RelSearchableNode, FloatProperty, ArrayProperty, \
    DateProperty, IntegerProperty


class TakenOn(StructuredRel):
    rel = 'taken_on'


class TakenIn(StructuredRel):
    rel = 'taken_in'


class TakenBy(StructuredRel):
    rel = 'taken_by'


class HasTag(StructuredRel):
    rel = 'has_tag'


class Near(StructuredRel):
    rel = 'near'


class HasTagScore(StructuredRel):
    rel = 'has_tag_score'


class HasPhotosTakenOn(StructuredRel):
    rel = 'has_photos_taken_on'


class User(RelSearchableNode):
    username = StringProperty(unique_index=True)


class Image(RelSearchableNode):
    hash_code = StringProperty(unique_index=True)
    size = IntegerProperty(default=0)
    date = RelationshipTo("Date", TakenOn.rel, model=TakenOn)
    user = RelationshipTo("User", TakenBy.rel, model=TakenBy)
    next_img = RelationshipTo("Image", Near.rel, model=Near, cardinality=One)
    prev_img = RelationshipFrom("Image", Near.rel, model=Near, cardinality=One)


class Tag(RelSearchableNode):
    name = StringProperty(unique_index=True)
    image = RelationshipFrom("Image", HasTag.rel, model=HasTag)

class Score(RelSearchableNode):
    value = FloatProperty(default=0.0)
    box = ArrayProperty(default=None)
    algorithm = StringProperty(index=True)
    dataset = StringProperty()
    tag = StringProperty()
    image = RelationshipFrom("Image", HasTagScore.rel, model=HasTagScore)


class Date(RelSearchableNode):
    date = DateProperty(unique_index=True)
    year = IntegerProperty(index=True)
    month = IntegerProperty(index=True)
    day = IntegerProperty(index=True)
    week_day = IntegerProperty(index=True)
    user = RelationshipFrom("User", HasPhotosTakenOn.rel, model=HasPhotosTakenOn)
    image = RelationshipFrom("Image", TakenOn.rel, model=TakenOn)


if __name__ == '__main__':
    # DEMO
    score_filter = Q(tag__in=['cloth'], algorithm__in=['ALL'],
                     dataset__in=['MSCOCO', 'ImageNet', 'MSCogServ', 'LSCTags', 'LSCCaption'])
    score_filter &= Q(value__gte=0.4) | Q(value=0.0)
    a = Score.filter(score_filter)
    print(a.cypher_query)
    print()

    score_filter = Score.filter(image__tag__name__in=['dog'])
    print(score_filter.cypher_query)
    print()

    # Images taken by User who's name == 'root'
    # Here `user` is a property of Image which defines the relationship between an image and an user
    a = Image.filter(user__username='root')
    a &= Q(hash_code='a')
    # Images connected with a Tag node with name='a' or the relationship between them has the property max_score with a value >= 10
    a &= (Q(tag=Tag.filter(~Q(name='a'))) | Q(rel_tag__max_score__gte=10))
    print(a.cypher_query)
    print()

    a = Image.filter(user__username="admin", date=Date.filter(year=2020))
    print(a.cypher_query)
    print()

    image_filter = Image.filter(Q(tag__name__in=['dog', 'person', 'car', 'tree']), user__username='root',
                                date=Date.filter(year=2020))
    print(image_filter.cypher_query)
    print()

    image_filter = Image.filter(Q(user__username='root') & Q(date=Date.filter(year=1)))
    print(image_filter.cypher_query)
    print()

    # You can filter images by a connected Node without declaring it in the Image class
    # e.g: Image does not have a declared relationship with Topic
    # But in the Topic class we have a defined relationship between it and Image
    # This way of use is not recommended, because there can be multiple defined relationships with Image in Topic
    a = Image.filter(Q(tag=Tag.filter(name='dog')))
    print(a.cypher_query)

    print()

    a = Image.filter(tag=Tag.filter(name__startswith='d')) & Image.filter(size__gte=30)
    # Order by a property of the current node, DESC
    a = a.order_by('-size')
    # or order by multiple properties at once
    a = a.order_by('-size', 'size')

    # You can define a list of properties `default_select` in the node class
    # If you don't apply any `select` filter, then QueryBuilder will select all properties in the `default_select`
    # If you indicate some property using `select` filter then the `default_select` will be ignored
    a = a.select("hash_code")
    a = a.distinct()
    print(a.cypher_query)
    print()

    a = Image.filter(next_img__hash_code='a')
    print(a.cypher_query)
    print()

    a = Image.filter(next_img=Image.filter(next_img__hash_code='a'), prev_img__tag__name='dog')
    print(a.cypher_query)
