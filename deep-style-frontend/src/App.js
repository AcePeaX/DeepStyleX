import React, { useState, useEffect } from "react";
import axios from "axios";
import { FiRefreshCw } from "react-icons/fi";
import { ClipLoader } from "react-spinners";
import "./App.css";

const host = "http://localhost:8080/"

const App = () => {
  const [models, setModels] = useState([]);
  const [device, setDevice] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [contentImage, setContentImage] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [generatedImages, setGeneratedImages] = useState([]);


  // Fetch models from the server
  const fetchModels = async () => {
    try {
      const response = await axios.get(host+"models", {timeout: 3000});
      setModels(response.data.models);
      setDevice(response.data.cuda ? "GPU" : "CPU")
    } catch (error) {
      console.error("Error fetching models:", error);
    }
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
      updatedImages[0].style = response.data.style_image_url;
      updatedImages[0].loading = false;
      setGeneratedImages(updatedImages);
    } catch (error) {
      console.error("Error processing image:", error);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  return (
    <div className="p-5 space-y-6">
      {/* Top Section */}
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold">DeepStyleX{device==="" ? "" : (" - "+device)}</h1>
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
                  alt="Generated"
                  className="object-cover style-image rounded center-self-style"
                />
              )}
              <img
                src={image.content}
                alt="Uploaded Content"
                className="object-cover initial-image rounded"
              />
              {image.loading ? (
                <div className="flex justify-center items-center h-32 bg-gray-100 rounded result-image image-loader-container">
                  <ClipLoader color="#000" size={40} />
                </div>
              ) : (
                <img
                  src={image.result}
                  alt="Generated"
                  className="object-cover rounded result-image"
                />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default App;
