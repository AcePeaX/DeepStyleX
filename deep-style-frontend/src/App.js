import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { FiRefreshCw } from "react-icons/fi";
import { ClipLoader } from "react-spinners";
import ImageNotFound from "./broken-image.png"
import Logo from "./logo.png"
import "./App.css";

const dev_host = "http://localhost:8080/"
const prod_host = ""

const host = process.env.NODE_ENV==='production' ? prod_host : dev_host 

const App = () => {
  const [models, setModels] = useState([]);
  const [device, setDevice] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [contentImage, setContentImage] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [maxResolution, setMaxResolution] = useState("");


  // Fetch models from the server
  const fetchModels = useCallback(async () => {
    try {
      const response = await axios.get(host+"models", {timeout: 3000});
      setModels(response.data.models);
      setDevice(response.data.cuda ? "GPU" : "CPU")
      if(maxResolution===""){
        setMaxResolution(response.data.cuda ? 0 : 600000)
      }
    } catch (error) {
      console.error("Error fetching models:", error);
    }
  },[maxResolution]);

  const openModal = (image) => {
    setSelectedImage(image);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedImage(null);
  };

  // Handle image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setContentImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  // Submit the image for processing
  const handleSubmit = async () => {
    if (!contentImage || !selectedModel) {
      alert("Please select a model and upload an image.");
      return;
    }

    const formData = new FormData();
    formData.append("content_image", contentImage);
    formData.append("model_name", selectedModel);
    formData.append("max_resolution", maxResolution);

    const newImage = {
      content: imagePreview,
      result: null,
      loading: true,
    };
    const newGeneratedImagesArray = [newImage, ...generatedImages]
    setGeneratedImages(newGeneratedImagesArray);

    try {
      const response = await axios.post(host+"process", formData, {timeout: 20000});
      const updatedImages = [...newGeneratedImagesArray];
      updatedImages[0].result = response.data.styled_image_url;
      if(response.data.style_image_url){
        updatedImages[0].style = response.data.style_image_url;
      }
      else{
        updatedImages[0].style = ImageNotFound;
      }
      updatedImages[0].loading = false;
      setGeneratedImages(updatedImages);
    } catch (error) {
      console.error("Error processing image:", error);
    }
  };

  useEffect(() => {
    fetchModels();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
    <div className="p-5 space-y-6">
      {/* Top Section */}
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold title-container">
            <div className="logo-container"><img src={Logo} alt="Logo"/></div>
            DeepStyleX{device==="" ? "" : (" - on "+device)}
          </h1>
          <button
            onClick={fetchModels}
            className="flex items-center bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            Reload Models <FiRefreshCw className="ml-2" />
          </button>
        </div>

        {/* Model Selector */}
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full border px-4 py-2 rounded"
        >
          <option value="">Select a Model</option>
          {models.map((model, index) => (
            <option key={index} value={model}>
              {model}
            </option>
          ))}
        </select>
        <div className="space-y-2 resol-selector">
          <label htmlFor="max-resolution" className="block font-semibold">
            Max Resolution (width x height) [0 for infinity]
          </label>
          <input
            id="max-resolution"
            type="number"
            placeholder="Enter max resolution"
            value={maxResolution}
            onChange={(e) => setMaxResolution(e.target.value)}
            className="border px-4 py-2 rounded w-full"
          />
        </div>
        {/* Image Upload */}
        <div className="space-y-2">
          <label
            htmlFor="file-upload"
            className="cursor-pointer bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 upload-button"
          >
            Upload Image
          </label>
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="hidden"
          />
          {imagePreview && (
            <div className="mt-2">
              <img
                src={imagePreview}
                alt="Preview"
                className="w-32 h-32 object-cover rounded shadow-lg"
              />
            </div>
          )}
        </div>

        {/* Submit Button */}
        <button
          onClick={handleSubmit}
          className={"bg-green-500 text-white px-4 py-2 rounded"+((selectedModel==="" || contentImage==null) ? " bg-gray-500" : " bg-green-500 hover:bg-green-600")}
          disabled={(selectedModel==="" || contentImage==null)}
        >
          Submit
        </button>
      </div>

      {/* Bottom Section */}
      <div className="space-y-4">
        <h2 className="text-xl font-bold">Generated Images</h2>
        <div className="list-container">
          {generatedImages.map((image, index) => (
            <div key={index} className="result-container">
              {image.loading ? (
                <div className="flex style-image justify-center items-center h-32 bg-gray-100 rounded style-image image-loader-container-style">
                  <ClipLoader color="#000" size={40} />
                </div>
              ) : (
                <img
                  src={image.style}
                  onClick={()=>{openModal(image.style)}}
                  alt="Generated"
                  className="object-cover style-image rounded center-self-style clickable"
                />
              )}
              <img
                src={image.content}
                onClick={()=>{openModal(image.content)}}
                alt="Uploaded Content"
                className="object-cover initial-image rounded clickable"
              />
              {image.loading ? (
                <div className="flex justify-center items-center h-32 bg-gray-100 rounded result-image image-loader-container">
                  <ClipLoader color="#000" size={40} />
                </div>
              ) : (
                <img
                  src={image.result}
                  onClick={()=>{openModal(image.result)}}
                  alt="Generated"
                  className="object-cover rounded result-image clickable"
                />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>

    {isModalOpen && (
      <div className="modal" onClick={closeModal}>
        <div className="modal-content" onClick={(e) => e.stopPropagation()}>
          <img src={selectedImage} alt="Full View" className="full-image" />
          <button onClick={closeModal} className="close-button">Close</button>
        </div>
      </div>
    )}
    </>
  );
};

export default App;
