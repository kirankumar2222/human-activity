from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):

    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender = models.CharField(max_length=30)
    address = models.CharField(max_length=30)


class early_hosp_prediction(models.Model):

    pid= models.CharField(max_length=300)
    gender= models.CharField(max_length=300)
    age= models.CharField(max_length=300)
    bp= models.CharField(max_length=300)
    hb= models.CharField(max_length=300)
    Year= models.CharField(max_length=300)
    facility_Id= models.CharField(max_length=300)
    facility_Name= models.CharField(max_length=300)
    APR_DRG_Code= models.CharField(max_length=300)
    APR_Severity_of_Illness_code= models.CharField(max_length=300)
    APR_DRG_Desc= models.CharField(max_length=300)
    APR_Severity_of_Illness_Desc= models.CharField(max_length=300)
    APR_MSC= models.CharField(max_length=300)
    APR_MSD= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



