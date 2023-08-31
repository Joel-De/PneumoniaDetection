"use client";

import {
  Flex,
  Button,
  Icon,
  Image,
  HStack,
  Spacer,
  Center,
} from "@chakra-ui/react";
import LoginModal from "./login_modal";

import { AiOutlineUserAdd, AiOutlineUser } from "react-icons/ai";

export default function NavBar({
  isSignup,
  setIsSignup,
  isLoginOpen,
  onLoginOpen,
  onLoginClose,
}) {
  return (
    <>
      <LoginModal
        {...{
          isOpen: isLoginOpen,
          onClose: onLoginClose,
          isSignup: isSignup,
          setIsSignup: setIsSignup,
        }}
      />
      <Flex
        minWidth="max-content"
        alignItems="center"
        gap="4"
        width="100%"
        height="80px"
      >
        <HStack>{/* <Image src="./logo.png" height="80px" /> */}</HStack>

        <Spacer />
        <Spacer />
        <HStack position="fixed" marginLeft="85%" marginRight="100px">
          <Button
            bgColor="green"
            borderRadius="20px"
            leftIcon={<Icon as={AiOutlineUserAdd} />}
            onClick={(e) => {
              setIsSignup(false);
              onLoginOpen();
            }}
          >
            Login
          </Button>
          <Spacer />
          <Button
            bgColor="teal"
            borderRadius="20px"
            leftIcon={<Icon as={AiOutlineUser} />}
            onClick={(e) => {
              setIsSignup(true);
              onLoginOpen();
            }}
          >
            Sign Up
          </Button>
        </HStack>
      </Flex>
    </>
  );
}
