"use client";
import logo from "./logo.svg";
import "./App.css";

import { Fade, ScaleFade, Slide, SlideFade, Collapse } from "@chakra-ui/react";
import { Heading } from "@chakra-ui/react";
import { Grid, GridItem } from "@chakra-ui/react";
import { Spacer } from "@chakra-ui/react";
import { ChakraProvider } from "@chakra-ui/react";
import { Link } from "@chakra-ui/react";
import { ImGithub } from "react-icons/im";

import { AiOutlineCloudUpload, AiOutlineDisconnect } from "react-icons/ai";
import { Icon } from "@chakra-ui/react";


import {
  Box,
  Flex,
  Avatar,
  HStack,
  Text,
  IconButton,
  Button,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  MenuDivider,
  useDisclosure,
  useColorModeValue,
  Popover,
  PopoverTrigger,
  PopoverContent,
  Stack,
} from "@chakra-ui/react";
import { useInViewport } from "react-in-viewport";
import { useRef } from "react";

import NavBar from "./nav.js";

import {
    
    HamburgerIcon, CloseIcon,
    ChevronDownIcon,
    ChevronRightIcon,
  } from '@chakra-ui/icons'



function App() {
  const ref = useRef(null);
  const { inViewport } = useInViewport(
    ref,
    { rootMargin: "-50px" },
    { disconnectOnLeave: false },
    {}
  );

  const linkColor = useColorModeValue('gray.600', 'gray.200')
  const linkHoverColor = useColorModeValue('gray.800', 'white')
  const popoverContentBgColor = useColorModeValue('white', 'gray.800')


  const { isOpen, onToggle } = useDisclosure()


  return (
    <ChakraProvider>
      <div className="App">
        <header className="App-header">
        
            <NavBar/>

          <Grid
            templateColumns="repeat(12, 1fr)"
            templateRows="repeat(7, 1fr)"
            gap={6}
          >
            <GridItem rowStart={3} colStart={2} rowSpan={1} colSpan={4}>
              <Box borderRadius="lg">
                <Heading textAlign={"left"} color={"#352F44"}>
                  PneumoniaDetect
                </Heading>
                <></>

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
                    <Button
                      leftIcon={<Icon as={ImGithub} />}
                      colorScheme="blue"
                    >
                      View Source Code
                    </Button>
                  </Link>
                </HStack>
              </Box>
            </GridItem>

            <GridItem rowStart={6} colStart={2} rowSpan={1} colSpan={4}>
              <ScaleFade initialScale={0.5} in={inViewport} >

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
                    <Button
                      leftIcon={<Icon as={ImGithub} />}
                      colorScheme="blue"
                    >
                      View Source Code
                    </Button>
                  </Link>
                </HStack>
              </Box>

              </ScaleFade>
            </GridItem>
          </Grid>
        </header>
      </div>
    </ChakraProvider>
  );
}

export default App;
