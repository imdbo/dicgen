# Generated by Django 3.0.7 on 2020-07-12 22:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dictionary', '0009_auto_20200713_0039'),
    ]

    operations = [
        migrations.AddField(
            model_name='context_token',
            name='similarity',
            field=models.FloatField(default=1),
            preserve_default=False,
        ),
    ]