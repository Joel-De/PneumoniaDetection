import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  Button,
  useDisclosure,
  Center,
  VStack,
  Input,
  InputGroup,
  InputRightElement,
  SlideFade,
  Collapse,
  Text,
} from "@chakra-ui/react";

import { ProductAPI } from "./api_layer";

import { useState } from "react";
import { Link } from "react-router-dom";

function PasswordInput({ setInput, isInvalid }) {
  const [show, setShow] = useState(false);
  const handleClick = () => setShow(!show);

  return (
    <InputGroup size="md">
      <Input
        isInvalid={isInvalid}
        pr="4.5rem"
        type={show ? "text" : "password"}
        placeholder="password"
        onChange={(e) => setInput(e.target.value)}
        errorBorderColor="crimson"
      />
      <InputRightElement width="4.5rem">
        <Button h="1.75rem" size="sm" onClick={handleClick}>
          {show ? "Hide" : "Show"}
        </Button>
      </InputRightElement>
    </InputGroup>
  );
}

export default function LoginModal({ isOpen, onClose, isSignup, setIsSignup }) {
  const [username, setUsername] = useState("");
  const [isUsernameInvalid, setIsUsernameInvalid] = useState(false);
  const [password, setPassword] = useState("");
  const [isPasswordInvalid, setIsPasswordInvalid] = useState(false);
  const [firstname, setFirstname] = useState("");
  const [lastname, setLastname] = useState("");
  const [city, setCity] = useState("");
  const [userMessage, setUserMessage] = useState("");

  return (
    <>
      <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Login & Signup</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Center height="300px">
              <VStack>
                <Input
                  placeholder="username"
                  size="md"
                  onChange={(e) => setUsername(e.target.value)}
                  errorBorderColor="crimson"
                  isInvalid={isUsernameInvalid}
                />
                <PasswordInput
                  {...{ setInput: setPassword, isInvalid: isPasswordInvalid }}
                />

                {isSignup && (
                  <VStack width="100%">
                    <Input
                      placeholder="firstname"
                      size="md"
                      onChange={(e) => setFirstname(e.target.value)}
                    />
                    <Input
                      placeholder="lastname"
                      size="md"
                      onChange={(e) => setLastname(e.target.value)}
                    />
                    <Input
                      placeholder="city"
                      size="md"
                      onChange={(e) => setCity(e.target.value)}
                    />
                  </VStack>
                )}

                <Text>{userMessage}</Text>
                <Link
                  colorScheme="blue"
                  mr={3}
                  onClick={(e) => {
                    setIsSignup(!isSignup);
                  }}
                >
                  {isSignup
                    ? "Already have an account?"
                    : "Don't have an account?"}
                </Link>
              </VStack>
            </Center>
          </ModalBody>

          <ModalFooter>
            <Center width="100%">
              {isSignup ? (
                <Button
                  colorScheme="blue"
                  onClick={(e) => {
                    ProductAPI.createAccount(
                      username,
                      password,
                      firstname,
                      lastname,
                      city
                    )
                      .then(function (data) {
                        console.log(data);
                        setIsPasswordInvalid(false);
                        setUserMessage("Account created!");
                      })
                      .catch(function (error) {
                        console.log(error);
                        if (error.response) {
                          setIsPasswordInvalid(true);
                          setUserMessage(error.response.data.message);
                          console.log(error.response);
                        }
                      });
                  }}
                >
                  Create Account
                </Button>
              ) : (
                <Button
                  colorScheme="blue"
                  onClick={(e) => {
                    ProductAPI.login(username, password)
                      .then(function (data) {
                        console.log("Logged in!");
                        console.log(data);
                        console.log(data.data.cookie);
                        setIsPasswordInvalid(false);
                        setIsUsernameInvalid(false);
                        window.location.href = "/portal";
                      })
                      .catch(function (error) {
                        if (error.response) {
                          console.log(error.response);
                          if (
                            error.response.data.message == "Incorrect Username"
                          ) {
                            setIsUsernameInvalid(true);
                            setIsPasswordInvalid(false);
                          } else if (
                            error.response.data.message == "Incorrect Password"
                          ) {
                            setIsPasswordInvalid(true);
                            setIsUsernameInvalid(false);
                          }
                        }
                      });
                  }}
                >
                  Login
                </Button>
              )}
            </Center>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </>
  );
}
