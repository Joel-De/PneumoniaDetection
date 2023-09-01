import datetime
import uuid

import psycopg
import asyncio
import sys
from pydantic import BaseModel
from psycopg.rows import dict_row
from typing import Optional
from psycopg import DatabaseError
from enum import Enum
import constants


class Doctor(BaseModel):
    username: str
    password: str  # this is a hashed password NOT plaintext
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
    patient_image: Optional[bytes] = None
    creation_date: datetime.datetime


class Diagnosis(BaseModel):
    patient_uuid: uuid.UUID
    doctor_uuid: uuid.UUID
    diagnosis: bool
    creation_date: datetime.datetime
    diagnosis_uuid: Optional[uuid.UUID] = None


def getConnectionString():
    return f"dbname={constants.DATABASE_NAME} user={constants.DATABASE_USER} password={constants.DATABASE_PASSWORD} host={constants.DATABASE_URL}"


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
                    creation_date timestamp,
                    patient_image bytea)
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


async def addDoctor(doctor: Doctor) -> Doctor:
    async with await psycopg.AsyncConnection.connect(
        getConnectionString(), row_factory=dict_row
    ) as conn:
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


async def getAllPatients(doctor_uuid: uuid.UUID) -> list[Patient]:
    async with await psycopg.AsyncConnection.connect(
        getConnectionString(), row_factory=dict_row
    ) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT first_name ,
                    last_name ,
                    age ,
                    sex ,
                    health_card_number  ,
                    patient_uuid ,
                    creation_date 
                FROM patients 
                WHERE patient_uuid IN (SELECT UNNEST("patient_ids") FROM  users WHERE doctor_uuid=(%s));
                """,
                (doctor_uuid,),
            )

            if (queryResults := await cur.fetchall()) is None:
                return []

            return [Patient.model_validate(queryResult) for queryResult in queryResults]


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

            patient = Patient.model_validate(queryResult)

            return patient


async def getPatientProfilePicture(healthCardNumber: str) -> Patient:
    async with await psycopg.AsyncConnection.connect(
        getConnectionString(), row_factory=dict_row
    ) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT patient_image FROM patients  WHERE health_card_number=%s
                """,
                (healthCardNumber,),
            )
            if (queryResult := await cur.fetchone()) is None:
                return None

            return queryResult["patient_image"]


async def addPatient(patient: Patient) -> Patient:
    async with await psycopg.AsyncConnection.connect(
        getConnectionString(), row_factory=dict_row
    ) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                            INSERT INTO patients (
                            first_name,
                            last_name,
                            age,
                            sex,
                            health_card_number,
                            creation_date,
                            patient_image
                            ) 
                            VALUES (%(first_name)s, %(last_name)s, %(age)s, %(sex)s, %(health_card_number)s, %(creation_date)s, %(patient_image)s)
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

            if (patientList := await cur.fetchone()) is None:
                raise DatabaseError("Doctor doesn't exist!")

            patientList = patientList[0]

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

            return patientList


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


async def getAllDiagnosis(doctorUUID: uuid.UUID = None, patientUUID: uuid.UUID = None):
    assert doctorUUID or patientUUID
    async with await psycopg.AsyncConnection.connect(
        getConnectionString(), row_factory=dict_row
    ) as conn:
        async with conn.cursor() as cur:
            if doctorUUID and patientUUID:
                await cur.execute(
                    f"""
                    SELECT * FROM diagnosis WHERE doctor_uuid=%s AND patient_uuid=%s
                    """,
                    (doctorUUID, patientUUID),
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

            return [
                Diagnosis.model_validate(queryResult) for queryResult in queryResults
            ]


if __name__ == "__main__":
    # for windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(initDB())
