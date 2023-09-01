"use client";
import "./App.css";

import {
  Center,
  ChakraProvider,
  Heading,
  Icon,
  Image,
  Link,
  Spacer,
  VStack,
} from "@chakra-ui/react";
import { useState } from "react";
import { AiOutlineCloudUpload } from "react-icons/ai";
import { ImGithub } from "react-icons/im";

import {
  Box,
  Button,
  Flex,
  HStack,
  Text,
  useDisclosure,
} from "@chakra-ui/react";

import NavBar from "./nav.js";

export default function HomePage() {
  const [isSignup, setIsSignup] = useState(false);
  const {
    isOpen: isLoginOpen,
    onOpen: onLoginOpen,
    onClose: onLoginClose,
  } = useDisclosure();

  return (
    <ChakraProvider>
      <div className="App">
        <header className="App-header">
          <NavBar
            {...{
              isSignup,
              setIsSignup,
              isLoginOpen,
              onLoginOpen,
              onLoginClose,
            }}
          />
          <Box
            borderRadius="lg"
            marginTop={"200px"}
            marginLeft={"130px"}
            marginBottom={"400px"}
          >
            <Heading textAlign={"left"} color={"#352F44"}>
              PneumoniaDetect
            </Heading>

            <Text textAlign={"left"} color={"#352F44"} fontSize={20}>
              Using AI to power
            </Text>
            <Text textAlign={"left"} color={"#352F44"} fontSize={20}>
              cutting edge healthcare.
            </Text>
            <Spacer />
            <Spacer />

            <HStack>
              <Button
                leftIcon={<Icon as={AiOutlineCloudUpload} />}
                colorScheme="blue"
                width={100}
              >
                Try Me!
              </Button>
              <Link href="https://github.com/Joel-De/PneumoniaDetection">
                <Button leftIcon={<Icon as={ImGithub} />} colorScheme="blue">
                  View Source Code
                </Button>
              </Link>
            </HStack>
          </Box>

          {/* <ScaleFade initialScale={0.5} in={inViewport}>
              <Box borderRadius="lg">
                <HStack>
                  <Button
                    leftIcon={<Icon as={AiOutlineCloudUpload} />}
                    colorScheme="blue"
                    width={100}
                  >
                    Try Me!
                  </Button>
                  <Link href="https://github.com/Joel-De/PneumoniaDetection">
                    <Button leftIcon={<Icon as={ImGithub} />} colorScheme="blue">
                      View Source Code
                    </Button>
                  </Link>
                </HStack>
              </Box>
            </ScaleFade> */}
        </header>
      </div>

      <Flex
        minWidth="max-content"
        alignItems="center"
        gap="4"
        width="100%"
        height="400px"
        bg="#26255A"
      >
        <Center width="100%">
          <HStack spacing="100px">
            <Text fontSize="6xl" fontWeight="bold" color="white">
              How it Works?
            </Text>
            <Text fontSize="2xl" width="500px" color="white">
              Using cutting edge image recognition models and thousands of
              X-Rays we’re able to Train AI capable of identifying cases of
              Pneumonia.
            </Text>
            <Image src="./ai_image.png" height="250px" />
          </HStack>
        </Center>
      </Flex>

      <Flex
        minWidth="max-content"
        alignItems="center"
        gap="4"
        width="100%"
        height="400px"
        bg="#5B59D9"
      >
        <Center width="100%">
          <HStack spacing="100px">
            <Image src="./trend_image.png" height="250px" />

            <Text fontSize="2xl" width="500px" color="white">
              We provide a secure solution for doctors to to manage automatic
              Pneumonia detection for any number of patients, keeping track of
              individual patient progress in easy to use tool.
            </Text>
            <Text fontSize="6xl" fontWeight="bold" color="white">
              What We Provide
            </Text>
          </HStack>
        </Center>
      </Flex>

      <Flex
        minWidth="max-content"
        alignItems="center"
        gap="4"
        width="100%"
        height="400px"
        bg="white"
      >
        <Center width="100%">
          <VStack>
            <Text fontSize="6xl" fontWeight="bold">
              Try it Today
            </Text>
            <Text fontSize="6xl" fontWeight="bold">
              for Free
            </Text>
            <Button
              width="110px"
              colorScheme="blue"
              onClick={() => {
                setIsSignup(true);
                onLoginOpen();
              }}
            >
              Sign up
            </Button>
          </VStack>
        </Center>
      </Flex>

      <Flex
        minWidth="max-content"
        alignItems="center"
        gap="4"
        width="100%"
        height="400px"
        bg="#EDEDFF"
      >
        <Center width="100%" margin="100px">
          <VStack width="100%">
            <Text fontSize="4xl" fontWeight="bold">
              Built Using Trusted Technologies
            </Text>

            <Flex w="100%">
              <Image height="175px" src="./docker_logo.png" />
              <Spacer />
              <Image height="175px" src="./redis_logo.png" />
              <Spacer />
              <Image height="175px" src="./fastapi_logo.png" />
              <Spacer />
              <Image height="175px" src="./react_logo.png" />
              <Spacer />
              <Image height="175px" src="./postgres_logo.png" />
            </Flex>
          </VStack>
        </Center>
      </Flex>
    </ChakraProvider>
  );
}
