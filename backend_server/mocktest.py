import random
import uuid

import requests
from backend_server.main_api import APIBodys, APIResponse


URL = "http://localhost:8000/"

names = ["John",
"William",
"James",
"Charles",
"George",
"Frank",
"Joseph",
"Thomas",
"Henry",
"Robert",
"Edward",
"Harry",
"Walter",
"Arthur",
"Fred",
"Albert",
"Samuel",
"David",
"Louis",
"Joe",
"Charlie",
"Clarence",
]



def randomGender():
    return random.sample(["Male", "Female"], 1)[0]


def runDiagnosis():
    r = requests.post(
        url=f"{URL}run_diagnosis", params=cookie | {"patient_uuid": patientUUID},
        files={"file": open(r"/confusion_matrix.png", 'rb').read()}
    )
    print(r.json())


r = requests.post(
    url=f"{URL}create_user/",
    json=APIBodys.Doctor(
        username="username1",
        password="tester1234",
        location="Canada",
        first_name="Jane",
        last_name="Doe",
    ).__dict__,
)
print(r.json())


r = requests.post(
    url=f"{URL}login", data={"username": "username3", "password": "tester1234"}
)
doctorCookie = APIResponse.model_validate(r.json()).data['cookie']


cookies = {'sessionID': doctorCookie}






cookie = {"cookie":doctorCookie}
for name in names:
    r = requests.post(
        url=f"{URL}add_patient", cookies=cookies, json=APIBodys.Patient(
            sex=randomGender(),
          age=random.randint(5,90),
          health_card_number=str(uuid.uuid4().hex),
            first_name=name,
            last_name=random.sample(names, 1)[0],
            patient_image=random.sample(names, 1)[0],
        ).__dict__
    )
    patientUUID = APIResponse.model_validate(r.json()).data["patient_uuid"]
    print(patientUUID)


runDiagnosis()
runDiagnosis()

r = requests.post(
    url=f"{URL}get_all_diagnosis", cookies=cookie, params = {"patient_uuid":patientUUID}, files={"file": open(
        r"/confusion_matrix.png", 'rb').read()}
)
results = APIResponse.model_validate(r.json())
[print(x) for x in results.data]




# r = requests.post(
#     url=f"{URL}remove_patient",  cookies=cookie, params = {"patient_uuid":patientUUID}
# )
# results = APIResponse.model_validate(r.json())
# print(results)

