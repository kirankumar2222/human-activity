from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,early_hosp_prediction,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Early_Hospitalization_Type(request):
    if request.method == "POST":

        if request.method == "POST":


            pid=request.POST.get('pid')
            gender=request.POST.get('gender')
            age=request.POST.get('age')
            bp=request.POST.get('bp')
            hb=request.POST.get('hb')
            Year=request.POST.get('Year')
            facility_Id=request.POST.get('facility_Id')
            facility_Name=request.POST.get('facility_Name')
            APR_DRG_Code=request.POST.get('APR_DRG_Code')
            APR_Severity_of_Illness_code=request.POST.get('APR_Severity_of_Illness_code')
            APR_DRG_Desc=request.POST.get('APR_DRG_Desc')
            APR_Severity_of_Illness_Desc=request.POST.get('APR_Severity_of_Illness_Desc')
            APR_MSC=request.POST.get('APR_MSC')
            APR_MSD=request.POST.get('APR_MSD')


        df = pd.read_csv('Datasets.csv')

        def apply_response(Label):
            if (Label == 0):
                return 0
            elif (Label == 1):
                return 1

        df['results'] = df['Label'].apply(apply_response)

        cv = CountVectorizer()
        X = df['pid']
        y = df['results']

        print("PID")
        print(X)
        print("Results")
        print(y)

        cv = CountVectorizer()
        X = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Extra Tree Classifier")
        from sklearn.tree import ExtraTreeClassifier
        etc_clf = ExtraTreeClassifier()
        etc_clf.fit(X_train, y_train)
        etcpredict = etc_clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, etcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, etcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, etcpredict))
        models.append(('RandomForestClassifier', etc_clf))
        detection_accuracy.objects.create(names="Extra Tree Classifier", ratio=accuracy_score(y_test, etcpredict) * 100)

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))
        models.append(('DecisionTreeClassifier', dtc))
        detection_accuracy.objects.create(names="Decision Tree Classifier",
                                          ratio=accuracy_score(y_test, dtcpredict) * 100)

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))
        detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))
        detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

        print("Gradient Boosting Classifier")

        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
            X_train,
            y_train)
        clfpredict = clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, clfpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, clfpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, clfpredict))
        models.append(('GradientBoostingClassifier', clf))
        detection_accuracy.objects.create(names="Gradient Boosting Classifier",
                                          ratio=accuracy_score(y_test, clfpredict) * 100)

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        pid1 = [pid]
        vector1 = cv.transform(pid1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if (prediction == 0):
            val = 'No Early Hospitalization and Charged High Cost'
        elif (prediction == 1):
            val = 'Early Hospitalization and Charged Less Cost'

        print(val)
        print(pred1)

        early_hosp_prediction.objects.create(
        pid=pid,
        gender=gender,
        age=age,
        bp=bp,
        hb=hb,
        Year=Year,
        facility_Id=facility_Id,
        facility_Name=facility_Name,
        APR_DRG_Code=APR_DRG_Code,
        APR_Severity_of_Illness_code=APR_Severity_of_Illness_code,
        APR_DRG_Desc=APR_DRG_Desc,
        APR_Severity_of_Illness_Desc=APR_Severity_of_Illness_Desc,
        APR_MSC=APR_MSC,
        APR_MSD=APR_MSD,
        Prediction=val)

        return render(request, 'RUser/Predict_Early_Hospitalization_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Early_Hospitalization_Type.html')



