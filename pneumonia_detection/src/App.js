"use client";
import logo from "./logo.svg";
import "./App.css";

import {
  Fade,
  ScaleFade,
  Slide,
  SlideFade,
  Collapse,
  VStack,
} from "@chakra-ui/react";
import { Heading } from "@chakra-ui/react";
import { Grid, GridItem } from "@chakra-ui/react";
import { Spacer, Image, Center } from "@chakra-ui/react";
import { ChakraProvider } from "@chakra-ui/react";
import { Link } from "@chakra-ui/react";
import { ImGithub } from "react-icons/im";

import { AiOutlineCloudUpload, AiOutlineDisconnect } from "react-icons/ai";
import { Icon } from "@chakra-ui/react";
import HomePage from "./home_page.js";
import DoctorPortal from "./doctor_portal";
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
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import NavBar from "./nav.js";

import {
  HamburgerIcon,
  CloseIcon,
  ChevronDownIcon,
  ChevronRightIcon,
} from "@chakra-ui/icons";
import { Navigate, useNavigate, useLocation } from "react-router-dom";

function App() {
  const ref = useRef(null);
  const { inViewport } = useInViewport(
    ref,
    { rootMargin: "-50px" },
    { disconnectOnLeave: false },
    {}
  );

  const linkColor = useColorModeValue("gray.600", "gray.200");
  const linkHoverColor = useColorModeValue("gray.800", "white");
  const popoverContentBgColor = useColorModeValue("white", "gray.800");

  return (
    <ChakraProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/portal" element={<DoctorPortal />} />
          <Route path="/" element={<HomePage />} />
        </Routes>
      </BrowserRouter>
    </ChakraProvider>
  );
}

export default App;
