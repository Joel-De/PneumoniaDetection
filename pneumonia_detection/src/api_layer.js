import axios from "axios";
import { useNavigate } from "react-router-dom";
import { redirect } from "react-router-dom";

const baseURL = `${process.env.REACT_APP_BACKEND_URL}`;

export const api = axios.create({
  // withCredentials: true,
  baseURL: `${process.env.REACT_APP_BACKEND_URL}`,
});

const instance = axios.create({
  withCredentials: true,
});

instance.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response.status == 403) {
      if (window.location.href.includes("portal")) {
        console.log(window.location.href);
        console.log("Invalid credentials detected, redirecting to main page!");
        // window.location.href = "/";
      }
    }

    throw error;
  }
);

export const ProductAPI = {
  login: async function (usernamme, password) {
    console.log("Making Login request to server.");

    const response = await instance.post(
      `${baseURL}login`,
      { username: usernamme, password: password },
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    return response.data;
  },
  logout: async function () {
    console.log("Making Logout request to server.");

    const response = await instance.post(`${baseURL}logout`);

    return response.data;
  },

  addPatient: async function (
    firstname,
    lastname,
    sex,
    age,
    city,
    healthCardNumber,
    file
  ) {
    console.log("Adding patient to doctor");

    var formData = new FormData();
    formData.append("file", file);

    // formData.append("first_name", file);
    // formData.append("last_name", lastname);
    // formData.append("sex", sex);
    // formData.append("age", Number(age));
    // formData.append("city", city);
    // formData.append("health_card_number", healthCardNumber);

    const response = await instance.post(`${baseURL}add_patient`, formData, {
      params: {
        first_name: firstname,
        last_name: lastname,
        sex: sex,
        age: Number(age),
        city: city,
        health_card_number: healthCardNumber,
      },
    });

    console.log("Adding patient to docasdstor");

    return response.data;
  },

  createAccount: async function (
    username,
    password,
    firstname,
    lastname,
    city
  ) {
    console.log("Created account");

    const response = await instance.post(`${baseURL}create_user`, {
      username: username,
      password: password,
      first_name: firstname,
      last_name: lastname,
      location: city,
    });
    return response.data;
  },

  getAllPatients: async function () {
    const response = await instance.get(`${baseURL}get_patients`);

    return response.data;
  },
  runDiagnosis: async function (file, patiendUUID) {
    console.log("Running Diagnosis");
    console.log(file);

    var formData = new FormData();
    formData.append("file", file);

    const response = await instance.post(
      `${baseURL}run_diagnosis`,
      formData,
      { params: { patient_uuid: patiendUUID } },
      {
        headers: {
          Accept: "application/json",
          "Content-Type": "multipart/form-data",
        },
      }
    );
    return response.data;
  },
  deletePatient: async function (patiendUUID) {
    const response = await instance.post(`${baseURL}remove_patient`, null, {
      params: { patient_uuid: patiendUUID },
    });

    return response;
  },

  getPatientProfilePicture: async function (health_card_number) {
    const response = await instance.get(`${baseURL}patient_profile_picture`, {
      params: { health_card_number: health_card_number },
      responseType: "arraybuffer",
    });

    return response.data;
  },
};

// defining the cancel API object for ProductAPI
// const cancelApiObject = defineCancelApiObject(ProductAPI)
