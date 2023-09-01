"use client";
import "./App.css";

import {
  AvatarGroup,
  Image,
  Spacer,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from "@chakra-ui/react";
import { AiOutlineUser } from "react-icons/ai";
import { useNavigate } from "react-router-dom";
import { ProductAPI } from "./api_layer";

import { Avatar, Button, Flex, HStack, Icon } from "@chakra-ui/react";
import { FaDoorOpen } from "react-icons/fa";

import { IoMdSettings } from "react-icons/io";

import { PatientView } from "./patient_view.js";

import { AddPatient } from "./add_patient.js";

function PortalNavbar() {
  const navigate = useNavigate();

  return (
    <>
      <Flex
        alignItems="center"
        gap="4"
        width="100%"
        height="80px"
        bgColor="#D9D9D9"
      >
        <Image src="./logo.png" height="80px" />
        <Spacer />

        <HStack minWidth="300px">
          <Button
            bgColor="teal"
            borderRadius="20px"
            leftIcon={<Icon as={IoMdSettings} />}
            onClick={(e) => {}}
          >
            Settings
          </Button>
          <Button
            bgColor="teal"
            borderRadius="20px"
            leftIcon={<Icon as={FaDoorOpen} />}
            onClick={(e) => {
              localStorage.removeItem("sessionID");
              ProductAPI.logout()
                .then(function (data) {
                  console.log("Successfully logged out!");
                })
                .catch(function (error) {
                  console.log(error);
                });

              navigate("/");
            }}
          >
            Logout
          </Button>
          <AvatarGroup spacing="1rem">
            <Avatar bg="teal.500" icon={<AiOutlineUser fontSize="1.5rem" />} />
          </AvatarGroup>
        </HStack>
      </Flex>
    </>
  );
}

export default function DoctorPortal() {
  return (
    <div>
      <PortalNavbar />

      <Flex
        minWidth="max-content"
        alignItems="center"
        gap="4"
        //width="75%"
        // height="80px"
        // bgColor="#D9D9D9"
        marginTop="50px"
        marginLeft="10%"
        marginRight="10%"
      >
        <Tabs width="100%">
          <TabList width="100%">
            <Tab>Add Pateint</Tab>
            <Tab>View Patients</Tab>
          </TabList>

          <TabPanels>
            <TabPanel>
              <AddPatient />
            </TabPanel>
            <TabPanel widht="100%">
              <PatientView />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Flex>
    </div>
  );
}
