from django.urls import include, path
from rest_framework.urlpatterns import format_suffix_patterns
from rest_framework import routers
from . import views

router = routers.DefaultRouter()
router.register(r'lemma', views.lemma_list)

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.home, name='home'),
    path('search_lemma/', views.search_lemma, name='search_lemma'),
    path('rest/', include(router.urls)),
]
