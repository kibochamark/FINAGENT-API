from django.db import models

# Create your models here.



class Agent(models.Model):
    query = models.TextField()
    response = models.TextField()


    created_at = models.DateTimeField(auto_now=True, auto_created=True)
    updated_at = models.DateTimeField(auto_now=True)


    def __str__(self):
        return self.query