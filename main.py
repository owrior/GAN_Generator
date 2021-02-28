from create_distribution import Dgen
from preprocess import ProbTransform, OneHot

d = Dgen(2, 4, 1000)
pt = ProbTransform(['store', 'product'])
oh = OneHot('store')
d.generate_data()

x = d._df
print(x)

x = pt.fit_transform(x)

x = oh.fit_transform(x)


x = oh.inverse_transform(x)

x = pt.inverse_transform(x)
print(x)