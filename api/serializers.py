from rest_framework import serializers

from api.models import Agent


class AgentSerializer(serializers.ModelSerializer):

    class Meta:
        model=Agent
        fields="__all__"