# Generated by Django 4.0.4 on 2022-05-03 21:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('QuerySearch', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='autocomplete',
            name='last_name',
            field=models.CharField(blank=True, max_length=30, null=True),
        ),
        migrations.AlterField(
            model_name='autocomplete',
            name='middle_name',
            field=models.CharField(blank=True, max_length=30, null=True),
        ),
    ]