# Generated by Django 3.0.7 on 2020-07-16 08:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dictionary', '0012_auto_20200716_0923'),
    ]

    operations = [
        migrations.AlterField(
            model_name='lemma',
            name='global_pos_tag',
            field=models.ManyToManyField(to='dictionary.Pos_frequency'),
        ),
    ]