import datetime
import logging
import uuid

import psycopg
import asyncio
from pydantic import BaseModel
from psycopg.rows import dict_row
from typing import Optional
from psycopg import DatabaseError
from enum import Enum


class Doctor(BaseModel):
    username: str
    password: str # this is a hashed password NOT plaintext
    location: str
    first_name: str
    last_name: str
    patient_ids: list[uuid.UUID]
    creation_date: datetime.datetime
    doctor_uuid: Optional[uuid.UUID] = None


class Patient(BaseModel):
    first_name: str
    last_name: str
    sex: str
    age: int
    health_card_number: str
    patient_uuid: Optional[uuid.UUID] = None
    creation_date: datetime.datetime


class Diagnosis(BaseModel):
    patient_uuid: uuid.UUID
    doctor_uuid: uuid.UUID
    diagnosis: bool
    creation_date: datetime.datetime
    diagnosis_uuid: Optional[uuid.UUID] = None


def getConnectionString():
    return "dbname=postgres user=postgres password=test123"


async def initDB():
    async with await psycopg.AsyncConnection.connect(getConnectionString()) as conn:
        # Open a cursor to perform database operations
        async with conn.cursor() as cur:
            # Execute a command: this creates a new table


            if 1:

                await cur.execute(
                    """
                DROP SCHEMA public CASCADE;
                CREATE SCHEMA public;
                """
                )

            await cur.execute(
                """CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; 
                                 SELECT uuid_generate_v4();"""
            )

            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username text PRIMARY KEY,
                    password text,
                    patient_ids uuid[],
                    creation_date timestamp,
                    location text,
                    first_name text,
                    last_name text,
                    doctor_uuid uuid DEFAULT uuid_generate_v4())
                """
            )

            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    first_name text,
                    last_name text,
                    age text,
                    sex text,
                    health_card_number text PRIMARY KEY,
                    patient_uuid uuid DEFAULT uuid_generate_v4(),
                    creation_date timestamp)
                """
            )

            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnosis (
                    patient_uuid uuid,
                    doctor_uuid uuid,
                    diagnosis boolean,
                    creation_date timestamp,
                    diagnosis_uuid uuid DEFAULT uuid_generate_v4())
                """
            )

            await conn.commit()


async def removeDoctor(username: str) -> Doctor:
    async with await psycopg.AsyncConnection.connect(
        getConnectionString(), row_factory=dict_row
    ) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                DELETE FROM users WHERE username=%s RETURNING *;
                """,
                (username,),
            )

            queryResult = await cur.fetchone()
            doctor = Doctor.model_validate(queryResult)
            await conn.commit()
            return doctor


def updateUser():
    pass


async def addDoctor(doctor: Doctor) -> Doctor:
    async with await psycopg.AsyncConnection.connect(getConnectionString() , row_factory=dict_row) as conn:
        async with conn.cursor() as cur:

            await cur.execute(
                """
                INSERT INTO users (
                username,
                password,
                patient_ids,
                creation_date,
                location,
                first_name,
                last_name) 
                VALUES (%(username)s, %(password)s, %(patient_ids)s, %(creation_date)s, %(location)s, %(first_name)s, %(last_name)s)
                RETURNING *
                """,
                doctor.__dict__,
            )
            queryResult = await cur.fetchone()
            doctor = Doctor.model_validate(queryResult)
            await conn.commit()
            return doctor

async def getAllPatients(username: str) -> list[Patient]:
    async with await psycopg.AsyncConnection.connect(getConnectionString() , row_factory=dict_row) as conn:
        async with conn.cursor() as cur:


            await cur.execute(
                """
                SELECT * 
                FROM patients 
                WHERE patient_uuid IN (SELECT UNNEST("patient_ids") FROM  users WHERE username=(%s));
                """,
                (username,),
            )


            if (queryResults := await cur.fetchall()) is None:
                return []

            return [Patient.model_validate(queryResult) for queryResult in  queryResults]





async def getDoctor(username: str) -> Doctor | None:
    async with await psycopg.AsyncConnection.connect(
        getConnectionString(), row_factory=dict_row
    ) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT * FROM users  WHERE username=%s
                """,
                (username,),
            )
            if (queryResult := await cur.fetchone()) is None:
                return None

            doctor = Doctor.model_validate(queryResult)
            return doctor


async def getPatient(healthCardNumber: str) -> Patient:
    async with await psycopg.AsyncConnection.connect(
        getConnectionString(), row_factory=dict_row
    ) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT * FROM patients  WHERE health_card_number=%s
                """,
                (healthCardNumber,),
            )
            if (queryResult := await cur.fetchone()) is None:
                return None
            pateint = Patient.model_validate(queryResult)
            return pateint


async def addPatient(patient: Patient) -> Patient:
    async with await psycopg.AsyncConnection.connect(getConnectionString(), row_factory=dict_row) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                            INSERT INTO patients (
                            first_name,
                            last_name,
                            age,
                            sex,
                            health_card_number,
                            creation_date
                            ) 
                            VALUES (%(first_name)s, %(last_name)s, %(age)s, %(sex)s, %(health_card_number)s, %(creation_date)s)
                            RETURNING *
                            """,
                patient.__dict__,
            )

            queryResult = await cur.fetchone()
            patient = Patient.model_validate(queryResult)
            await conn.commit()
            return patient




async def addPatientToDoctor(patientUUID: uuid.UUID, doctorUUID: uuid.UUID):
    async with await psycopg.AsyncConnection.connect(getConnectionString()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT patient_ids FROM users WHERE doctor_uuid=%s
                """,
                (doctorUUID,),
            )

            if (patientList := (await cur.fetchone())[0]) is None:
                raise DatabaseError("Doctor doesn't exist!")

            if patientUUID in patientList:
                raise DatabaseError("Doctor already assigned this patient!")
            else:
                patientList.append(patientUUID)

            await cur.execute(
                """
                UPDATE users SET patient_ids=%s WHERE doctor_uuid=%s
                """,
                (patientList, doctorUUID),
            )

            return patientList


async def removePatientFromDoctor(patientUUID: uuid.UUID, doctorUUID: uuid.UUID):
    async with await psycopg.AsyncConnection.connect(getConnectionString()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT patient_ids FROM users  WHERE doctor_uuid=%s
                """,
                (doctorUUID,),
            )
            patientList = (await cur.fetchone())[0]

            if patientList is None:
                raise DatabaseError("Doctor doesn't exist!")

            if patientUUID not in patientList:
                raise DatabaseError("Patient not assigned to Doctor")
            else:
                patientList.remove(patientUUID)

            await cur.execute(
                """
                UPDATE users SET patient_ids=%s WHERE doctor_uuid=%s
                """,
                (patientList, doctorUUID),
            )

            return doctor


async def addDiagnosis(diagnosis: Diagnosis):
    async with await psycopg.AsyncConnection.connect(getConnectionString()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT * from patients WHERE patient_uuid=%s 
                """,
                (diagnosis.patient_uuid,),
            )
            if (await cur.fetchone()) is None:
                raise DatabaseError(f"Missing patient {diagnosis.patient_uuid}")

            await cur.execute(
                """
                INSERT INTO diagnosis (
                patient_uuid,
                doctor_uuid,
                diagnosis,
                creation_date
                ) 
                VALUES (%(patient_uuid)s, %(doctor_uuid)s, %(diagnosis)s, %(creation_date)s)
                """,
                diagnosis.__dict__,
            )

            await conn.commit()


class uuidTypes(Enum):
    DOCTOR = 1
    PATIENT = 2
    DIAGNOSIS = 3


async def getDiagnosis(
    uuid: uuid.UUID, uuidType: uuidTypes = uuidTypes.DIAGNOSIS
) -> Diagnosis | None:
    async with await psycopg.AsyncConnection.connect(
        getConnectionString(), row_factory=dict_row
    ) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT * FROM diagnosis WHERE {uuidType.name.lower()}_uuid=%s
                """,
                (uuid,),
            )

            if (queryResult := await cur.fetchone()) is None:
                return None
            diagnosis = Diagnosis.model_validate(queryResult)
            return diagnosis


async def getAllDiagnosis(doctorUUID:uuid.UUID = None, patientUUID:uuid.UUID = None):
    assert(doctorUUID or patientUUID)
    async with await psycopg.AsyncConnection.connect(
        getConnectionString(), row_factory=dict_row
    ) as conn:
        async with conn.cursor() as cur:
            if doctorUUID and patientUUID:
                await cur.execute(
                    f"""
                    SELECT * FROM diagnosis WHERE doctor_uuid=%s AND patient_uuid=%s
                    """,
                    (doctorUUID,patientUUID),
                )
            elif doctorUUID:
                await cur.execute(
                    f"""
                    SELECT * FROM diagnosis WHERE doctor_uuid=%s 
                    """,
                    (doctorUUID,),
                )
            else:
                await cur.execute(
                    f"""
                    SELECT * FROM diagnosis WHERE patient_uuid=%s
                    """,
                    (patientUUID,),
                )

            if (queryResults := await cur.fetchall()) is None:
                return []

            return [Diagnosis.model_validate(queryResult) for queryResult in queryResults]


if __name__ == '__main__':


    # For testing purposes
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(initDB())
    import hashlib
    doctorToAdd = Doctor(
        username="Doctor5",
        password=str(hashlib.sha256(b"test1234").hexdigest()),
        first_name="Jane",
        last_name="Doe",
        patient_ids=[],
        location="NYC",
        creation_date=datetime.datetime.now(),
    )

    asyncio.run(addDoctor(doctorToAdd))

    doctor = asyncio.run(getDoctor("Doctor5"))

    asyncio.run(
        addPatient(
            Patient(
                first_name="John",
                last_name="Doe",
                doctor_uuid=doctor.doctor_uuid,
                sex="Male",
                age=43,
                location="NYC",
                health_card_number="Healthcard1",
                creation_date=datetime.datetime.now(),
            )
        )
    )

    asyncio.run(
        addPatient(
            Patient(
                first_name="Jane",
                last_name="Doe",
                doctor_uuid=doctor.doctor_uuid,
                sex="Male",
                age=43,
                location="NYC",
                health_card_number="Healthcard14",
                creation_date=datetime.datetime.now(),
            )
        )
    )


    for i in range(10):
        asyncio.run(
            addPatient(
                Patient(
                    first_name="John",
                    last_name="Doe",
                    doctor_uuid=doctor.doctor_uuid,
                    sex="Male",
                    age=43,
                    location="NYC",
                    health_card_number=f"Healthcard{i}_1",
                    creation_date=datetime.datetime.now(),
                )
            )
        )


        patient1 = asyncio.run(getPatient(f"Healthcard{i}_1"))
        asyncio.run(addPatientToDoctor(patient1.patient_uuid, doctor.doctor_uuid))

    patient2 = asyncio.run(getPatient("Healthcard14"))
    asyncio.run(addPatientToDoctor(patient2.patient_uuid, doctor.doctor_uuid))
    asyncio.run(removePatientFromDoctor(patient2.patient_uuid, doctor.doctor_uuid))

    res = asyncio.run(getAllPatients("Doctor5"))
    for item in res:
        print(item)








    asyncio.run(
        addDiagnosis(
            Diagnosis(
                patient_uuid=patient2.patient_uuid,
                doctor_uuid=doctor.doctor_uuid,
                diagnosis=True,
                creation_date=datetime.datetime.now(),
            )
        )
    )

    asyncio.run(
        addDiagnosis(
            Diagnosis(
                patient_uuid=patient2.patient_uuid,
                doctor_uuid=doctor.doctor_uuid,
                diagnosis=True,
                creation_date=datetime.datetime.now(),
            )
        )
    )

    asyncio.run(
        addDiagnosis(
            Diagnosis(
                patient_uuid=patient2.patient_uuid,
                doctor_uuid=doctor.doctor_uuid,
                diagnosis=True,
                creation_date=datetime.datetime.now(),
            )
        )
    )




    ####
    res1 = asyncio.run(getDiagnosis(patient2.patient_uuid, uuidTypes.PATIENT))
    print(res1)
    res2 = asyncio.run(getDiagnosis(doctor.doctor_uuid, uuidTypes.DOCTOR))
    print(res2)
    assert res1 == res2
    ####


    res = asyncio.run(getAllDiagnosis(
        patient2.patient_uuid
    ))

    [print(x) for x in res]

    # print(f"Removed {res}")


