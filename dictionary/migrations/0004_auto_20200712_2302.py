# Generated by Django 3.0.7 on 2020-07-12 21:02

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dictionary', '0003_auto_20200712_2258'),
    ]

    operations = [
        migrations.RenameField(
            model_name='lemma',
            old_name='global_pos_tags',
            new_name='global_pos_tag',
        ),
    ]
