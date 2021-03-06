# Generated by Django 3.0.7 on 2020-07-12 22:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dictionary', '0008_context_tokens'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Context_tokens',
            new_name='Context_token',
        ),
        migrations.RemoveField(
            model_name='lemma',
            name='negative_lemmas',
        ),
        migrations.RemoveField(
            model_name='lemma',
            name='vector_lemmas',
        ),
        migrations.AddField(
            model_name='lemma',
            name='negative_lemma',
            field=models.ManyToManyField(blank=True, related_name='negative_lemma', to='dictionary.Context_token'),
        ),
        migrations.AddField(
            model_name='lemma',
            name='positive_lemma',
            field=models.ManyToManyField(blank=True, related_name='positive_lemma', to='dictionary.Context_token'),
        ),
        migrations.AlterField(
            model_name='lemma',
            name='disambiguations',
            field=models.ManyToManyField(blank=True, related_name='disambiguation_definition', to='dictionary.Definition'),
        ),
    ]
