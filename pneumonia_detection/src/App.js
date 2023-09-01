"use client";
import "./App.css";

import { ChakraProvider } from "@chakra-ui/react";

import { BrowserRouter, Route, Routes } from "react-router-dom";
import DoctorPortal from "./doctor_portal";
import HomePage from "./home_page.js";

function App() {
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
