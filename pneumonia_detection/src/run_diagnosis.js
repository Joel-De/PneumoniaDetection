"use client";
import "./App.css";

import { Center, Image } from "@chakra-ui/react";

import { ProductAPI } from "./api_layer";

import { Button, HStack } from "@chakra-ui/react";
import { useEffect, useState } from "react";

export function RunDiagnosis() {
  const [patientData, setPatientData] = useState([]);
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();

  useEffect(() => {
    ProductAPI.getAllPatients()
      .then(function (data) {
        if (data) {
          console.log("asdasdas");

          setPatientData(data.data);
          console.log(patientData);
        }
      })
      .catch(function (error) {
        console.log(error);
      });
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile.pictureAsFile);
    setPreview(objectUrl);

    // free memory when ever this component is unmounted
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const onSelectFile = (e) => {
    if (!e.target.files || e.target.files.length === 0) {
      setSelectedFile(undefined);
      return;
    }

    // I've kept this example simple by using the first image instead of multiple
    setSelectedFile({
      picturePreview: URL.createObjectURL(e.target.files[0]),
      pictureAsFile: e.target.files[0],
    });
  };

  return (
    <>
      <Center>
        <HStack spacing="100">
          <div>
            <Button type="file" name="file" onChange={onSelectFile}>
              tes
            </Button>

            <Button
              onClick={(e) => {
                ProductAPI.runDiagnosis(
                  selectedFile.pictureAsFile,
                  "0c74da98-053f-4dc9-8980-69e0094adef9",
                )
                  .then(function (data) {})
                  .catch(function (error) {
                    console.log(error);
                  });
              }}
            >
              Upload Scan
            </Button>
            {selectedFile && <img src={preview} width="250px" />}
            <Image src="" />
          </div>
        </HStack>
      </Center>
    </>
  );
}
