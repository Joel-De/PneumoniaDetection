"use client";
import "../App.css";

import {
  Card,
  CardBody,
  Center,
  Container,
  Image,
  Input,
  InputGroup,
  InputLeftElement,
  Spacer,
  VStack,
  WrapItem,
} from "@chakra-ui/react";

import { ProductAPI } from "../api_layer";

import {
  Avatar,
  Box,
  Button,
  Flex,
  HStack,
  Icon,
  Text,
} from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { AiFillEye, AiOutlineSearch } from "react-icons/ai";

import { BsFillTrashFill } from "react-icons/bs";
import { GrRefresh } from "react-icons/gr";
export function PatientView() {
  const [patientData, setPatientData] = useState([]);
  const [cardDisplay, setCardDisplay] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [patientProfilePicture, setPatientProfilePicture] = useState();
  const [preview, setPreview] = useState();
  const [diagnosis, setDiagnosis] = useState(-1);
  const [searchBarValue, setSearchBarValue] = useState("");

  function updatePatientList() {
    ProductAPI.getAllPatients()
      .then(function (data) {
        console.log("asdasdas");

        setPatientData(data.data);
        console.log(patientData);
      })
      .catch(function (error) {
        console.log(error);
      });
  }
  useEffect(() => {
    updatePatientList();
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile.pictureAsFile);
    setPreview(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const onSelectFile = (e) => {
    if (!e.target.files || e.target.files.length === 0) {
      setSelectedFile(undefined);
      return;
    }

    setSelectedFile({
      picturePreview: URL.createObjectURL(e.target.files[0]),
      pictureAsFile: e.target.files[0],
    });
  };

  return (
    <>
      <Center>
        <HStack spacing="75" maxWidth="100%">
          <VStack marginTop="20px" width="450px">
            <Text fontSize="4xl">Your Current Patient List</Text>
            <InputGroup>
              <InputLeftElement
                pointerEvents="none"
                children={<Icon as={AiOutlineSearch} />}
                size="sm"
              />
              <Input
                variant="outline"
                size="md"
                placeholder="Health Card Number"
                onChange={(e) => setSearchBarValue(e.target.value)}
              />
              <Button
                leftIcon={
                  <Center>
                    <Icon size="md" as={GrRefresh} />
                  </Center>
                }
                colorScheme="teal"
                onClick={() => {
                  updatePatientList();
                }}
              />
            </InputGroup>
            <Container maxHeight="200px" minH="600px" overflowY="auto">
              <VStack>
                {patientData
                  .filter(
                    (patient) =>
                      patient.health_card_number.startsWith(searchBarValue) ||
                      searchBarValue === "",
                  )
                  .map((patient) => {
                    return (
                      <Card variant="filled" width="100%">
                        <CardBody>
                          <Flex>
                            <div>
                              <Text>
                                Name: {patient.first_name} {patient.last_name}
                              </Text>
                              <Text>
                                Health Card Number: {patient.health_card_number}
                              </Text>
                            </div>
                            <Spacer />

                            <VStack>
                              <Button
                                onClick={(e) => {
                                  setDiagnosis(-1);
                                  setCardDisplay(patient);
                                  ProductAPI.getPatientProfilePicture(
                                    patient.health_card_number,
                                  ).then((res) => {
                                    const base64 = btoa(
                                      new Uint8Array(res).reduce(
                                        (data, byte) =>
                                          data + String.fromCharCode(byte),
                                        "",
                                      ),
                                    );
                                    setPatientProfilePicture(
                                      `data:image/png;base64,${base64}`,
                                    );
                                  });
                                }}
                                rightIcon={<Icon as={AiFillEye} />}
                              >
                                View
                              </Button>
                              <Button
                                onClick={(e) => {
                                  ProductAPI.deletePatient(patient.patient_uuid)
                                    .then(function (data) {
                                      updatePatientList();
                                    })
                                    .catch(function (error) {
                                      console.log(error);
                                    });
                                }}
                                rightIcon={<Icon as={BsFillTrashFill} />}
                              >
                                Delete
                              </Button>
                            </VStack>
                          </Flex>
                        </CardBody>
                      </Card>
                    );
                  })}
              </VStack>
            </Container>
          </VStack>

          <Card width="500px" height="500px">
            {cardDisplay !== null && (
              <CardBody>
                <Flex w="100%" flexDirection="column">
                  <Center>
                    <WrapItem>
                      <Avatar
                        size="2xl"
                        name={`${cardDisplay.firstname} ${cardDisplay.lastname}`}
                        src={patientProfilePicture}
                      />
                    </WrapItem>
                  </Center>

                  <Flex marginTop="10%" w="100%" flexDirection="column">
                    <Flex marginLeft="20px" marginRight="20px">
                      <div>
                        <Box
                          mt="1"
                          fontWeight="semibold"
                          as="h4"
                          lineHeight="tight"
                          noOfLines={1}
                        >
                          Firstname:
                        </Box>
                        <Box>{cardDisplay.first_name}</Box>
                      </div>

                      <Spacer />
                      <div>
                        <Box
                          mt="1"
                          fontWeight="semibold"
                          as="h4"
                          lineHeight="tight"
                          noOfLines={1}
                        >
                          Lastname:
                        </Box>
                        <Box>{cardDisplay.last_name}</Box>
                      </div>

                      <Spacer />
                      <div>
                        <Box
                          mt="1"
                          fontWeight="semibold"
                          as="h4"
                          lineHeight="tight"
                          noOfLines={1}
                        >
                          Age:
                        </Box>
                        <Box>{cardDisplay.age}</Box>
                      </div>
                    </Flex>

                    <Flex marginTop="40px" marginLeft="60px" marginRight="60px">
                      <div>
                        <Box
                          mt="1"
                          fontWeight="semibold"
                          as="h4"
                          lineHeight="tight"
                          noOfLines={1}
                        >
                          Health Card Number:
                        </Box>
                        <Box>{cardDisplay.health_card_number}</Box>
                      </div>
                      <Spacer />
                      <div>
                        <Box
                          mt="1"
                          fontWeight="semibold"
                          as="h4"
                          lineHeight="tight"
                          noOfLines={1}
                        >
                          Sex:
                        </Box>
                        <Box>{cardDisplay.sex}</Box>
                      </div>
                    </Flex>
                  </Flex>
                </Flex>

                <Flex marginTop="100px">
                  <Input
                    type="file"
                    name="file"
                    accept="image/*"
                    onChange={onSelectFile}
                  />
                  <Button
                    onClick={(e) => {
                      ProductAPI.runDiagnosis(
                        selectedFile.pictureAsFile,
                        cardDisplay.patient_uuid,
                      )
                        .then(function (data) {
                          console.log(data);
                          setDiagnosis(data.data.result);
                        })
                        .catch(function (error) {
                          console.log(error);
                        });
                    }}
                    isDisabled={selectedFile === null}
                  >
                    Upload Scan
                  </Button>
                </Flex>
              </CardBody>
            )}
          </Card>
          <Card width="500px">
            <CardBody>
              <Center>
                <Text fontSize="5xl">Scan Preview</Text>
              </Center>

              <Image
                width="500px"
                height="500px"
                src={preview}
                marginTop="20px"
              ></Image>
              <Center>
                {diagnosis === 0 && (
                  <Text fontSize="3xl">No Pneumonia detected.</Text>
                )}
                {diagnosis === 1 && (
                  <Text fontSize="3xl">
                    Pneumonia detected, please conduct further testing
                  </Text>
                )}
              </Center>
            </CardBody>
          </Card>
        </HStack>
      </Center>
    </>
  );
}
