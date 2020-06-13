# Generated by Django 3.0.7 on 2020-06-10 20:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dictionary', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Collocation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('collocation', models.CharField(max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='lemma',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('lemma', models.CharField(max_length=200)),
            ],
        ),
        migrations.DeleteModel(
            name='Top_common_lemmas',
        ),
        migrations.RenameField(
            model_name='definition',
            old_name='definition',
            new_name='definition_summary',
        ),
        migrations.AddField(
            model_name='definition',
            name='definition_vector',
            field=models.CharField(default=1, max_length=5000),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='definition',
            name='plural',
            field=models.CharField(default=1, max_length=20),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='definition',
            name='singular',
            field=models.CharField(default=1, max_length=20),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='lemma',
            name='definition',
            field=models.ManyToManyField(to='dictionary.Definition'),
        ),
        migrations.AddField(
            model_name='definition',
            name='collocations',
            field=models.ManyToManyField(to='dictionary.Collocation'),
        ),
    ]