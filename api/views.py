from django.shortcuts import render
from rest_framework import viewsets
from .models import Agent
from .serializers import AgentSerializer
from .agent import AgentExecuter
from asgiref.sync import sync_to_async
from rest_framework.response import Response
import asyncio





# Create your views here.
class AgentViewset(viewsets.ViewSet):
    serailizer_class = AgentSerializer()
    queryset= Agent.objects.all()


    def agent_query(self, request):
        query = request.data.get("query", None)

        if query is None:
            return Response({
                "error":"query should be available"
            }, status=400)



        agent=AgentExecuter()

        try:
            x,result = agent.query(query)
            print(result)
            serializer= AgentSerializer(data={
                "query":query,
                "response":result
            })

            print(serializer.is_valid())

            if serializer.is_valid(raise_exception=True):
                serializer.save()
                return Response({
                    "result":result
                }, x)
            else:
                return Response(serializer.errors, 400)
        except Exception as e:

            return Response({
                "error":e
            }, 400)


    def get_history(self, request):

        try:
            agent_query_responses = Agent.objects.all()

            serializer = AgentSerializer(agent_query_responses, many=True)

            return Response(serializer.data, 200)
        
        except Exception as e:

            return Response({
                "error":e
            }, 400)

