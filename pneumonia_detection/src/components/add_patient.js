"use client";
import "../App.css";

import {
  Alert,
  AlertIcon,
  Center,
  FormControl,
  FormHelperText,
  Input,
  Select,
  Stack,
  VStack,
  WrapItem,
} from "@chakra-ui/react";
import { ProductAPI } from "../api_layer";

import { Avatar, Button, HStack, Icon, Text } from "@chakra-ui/react";
import { useState } from "react";

import { AiOutlineUserAdd } from "react-icons/ai";

export function AddPatient() {
  const [sex, setSex] = useState("male");
  const [age, setAge] = useState(0);
  const [firstname, setFirstname] = useState("");
  const [lastname, setLastname] = useState("");
  const [city, setCity] = useState("");
  const [healthCardNumber, setHealthCardNumber] = useState("");
  const [addSuccess, setAddSuccess] = useState(false);
  const [patientProfilePicture, setPatientProfilePicture] = useState(null);

  const onSelectFile = (e) => {
    if (!e.target.files || e.target.files.length === 0) {
      setPatientProfilePicture(undefined);
      return;
    }

    setPatientProfilePicture({
      picturePreview: URL.createObjectURL(e.target.files[0]),
      pictureAsFile: e.target.files[0],
    });
  };

  return (
    <>
      <Center>
        <VStack width="50%">
          <Text fontSize="3xl">Configure Patient</Text>

          <HStack width="100%">
            <Stack width="50%">
              <FormControl variant="floating" id="first-name" isRequired>
                <Input
                  placeholder="Firstname"
                  onChange={(e) => setFirstname(e.target.value)}
                  isInvalid={firstname === ""}
                />
                <FormHelperText>Firstname</FormHelperText>
              </FormControl>
              <FormControl variant="floating" id="first-name" isRequired>
                <Input
                  placeholder="Lastname"
                  onChange={(e) => setLastname(e.target.value)}
                  isInvalid={lastname === ""}
                />
                <FormHelperText>Lastname</FormHelperText>
              </FormControl>
            </Stack>

            <Center width="50%">
              <WrapItem>
                <Avatar
                  size="2xl"
                  name={`${firstname} ${lastname}`}
                  src={
                    patientProfilePicture &&
                    patientProfilePicture.picturePreview
                  }
                />{" "}
              </WrapItem>
            </Center>
          </HStack>

          <HStack width="100%">
            <FormControl width="100%" isRequired>
              <Input
                placeholder="Age"
                onChange={(e) => setAge(e.target.value)}
                isInvalid={isNaN(age) || age === 0}
              />
              <FormHelperText>Patient age</FormHelperText>
            </FormControl>
            <FormControl variant="floating" isRequired>
              <Input
                placeholder="City"
                onChange={(e) => setCity(e.target.value)}
                isInvalid={city === ""}
              />
              <FormHelperText>City / Location</FormHelperText>
            </FormControl>
          </HStack>

          <FormControl variant="floating" isRequired>
            <Input
              placeholder="Health Card Number"
              onChange={(e) => setHealthCardNumber(e.target.value)}
              isInvalid={healthCardNumber === ""}
            />
            <FormHelperText>Health Card Number</FormHelperText>
          </FormControl>
          <FormControl variant="floating" isRequired>
            <Select onChange={(e) => setSex(e.target.value)}>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </Select>
            <FormHelperText>Gender</FormHelperText>
          </FormControl>

          <FormControl variant="floating" isRequired>
            <Input
              type="file"
              name="file"
              accept="image/*"
              onChange={onSelectFile}
            />
            <FormHelperText>Profile Picture</FormHelperText>
          </FormControl>

          <Button
            bgColor="teal"
            borderRadius="20px"
            leftIcon={<Icon as={AiOutlineUserAdd} />}
            onClick={(e) => {
              ProductAPI.addPatient(
                firstname,
                lastname,
                sex,
                age,
                city,
                healthCardNumber,
                patientProfilePicture.pictureAsFile,
              )
                .then(function (data) {
                  setAddSuccess(true);
                  setTimeout(() => {
                    setAddSuccess(false);
                  }, 3000);
                  console.log(data);
                })
                .catch(function (error) {
                  setAddSuccess(false);
                  console.log(error);
                });
            }}
            isDisabled={
              firstname === "" ||
              lastname === "" ||
              isNaN(age) ||
              age === 0 ||
              city === "" ||
              patientProfilePicture === null
            }
          >
            Add Patient
          </Button>
          {addSuccess && (
            <Alert status="success">
              <AlertIcon />
              Patient successfully added
            </Alert>
          )}
        </VStack>
      </Center>
    </>
  );
}
