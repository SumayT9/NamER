from django.shortcuts import render

# Create your views here.

from django.shortcuts import render
import json
import joblib
from django.contrib.auth.models import User #####
from django.http import JsonResponse , HttpResponse ####

from NamER.crf import perform_inference



# https://pypi.org/project/wikipedia/#description
def get_entities(request):
    text = request.GET.get('text', None)
    model1 = joblib.load("NamER/crf1_no_chunk")
    model2 = joblib.load("NamER/crf2_no_chunk")
    print("text", text)

    output = perform_inference(text, model1, model2)

    data = {
        'People' : output["People"],
        'Locations' : output["Locations"],
        "Organizations" : output["Organizations"],
        "Other" : output["Other"],
        'raw': 'Successful'
    }

    print('json-data to be sent: ', data)

    return JsonResponse(data)