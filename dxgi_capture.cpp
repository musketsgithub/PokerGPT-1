#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

namespace py = pybind11;

class DXGICapture {
public:
    DXGICapture() {
        D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &context);
        device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);
        dxgiDevice->GetParent(__uuidof(IDXGIAdapter), (void**)&dxgiAdapter);
        dxgiAdapter->EnumOutputs(0, &dxgiOutput);
        dxgiOutput->QueryInterface(__uuidof(IDXGIOutput1), (void**)&dxgiOutput1);
        dxgiOutput1->DuplicateOutput(device, &desktopDuplication);
    }

    py::array_t<uint8_t> capture_frame() {
        DXGI_OUTDUPL_FRAME_INFO frameInfo;
        IDXGIResource* desktopResource = nullptr;
        ID3D11Texture2D* acquiredImage = nullptr;

        if (FAILED(desktopDuplication->AcquireNextFrame(500, &frameInfo, &desktopResource)))
            return py::array_t<uint8_t>();  // Return empty array if failed

        desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&acquiredImage);

        D3D11_TEXTURE2D_DESC desc;
        acquiredImage->GetDesc(&desc);
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.MiscFlags = 0;

        ID3D11Texture2D* readableTexture;
        device->CreateTexture2D(&desc, nullptr, &readableTexture);
        context->CopyResource(readableTexture, acquiredImage);

        D3D11_MAPPED_SUBRESOURCE mappedResource;
        context->Map(readableTexture, 0, D3D11_MAP_READ, 0, &mappedResource);

        std::vector<uint8_t> imgData(desc.Width * desc.Height * 4);
        memcpy(imgData.data(), mappedResource.pData, imgData.size());

        context->Unmap(readableTexture, 0);
        readableTexture->Release();
        acquiredImage->Release();
        desktopResource->Release();
        desktopDuplication->ReleaseFrame();

        return py::array_t<uint8_t>({desc.Height, desc.Width, 4}, imgData.data());
    }

private:
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    IDXGIDevice* dxgiDevice = nullptr;
    IDXGIAdapter* dxgiAdapter = nullptr;
    IDXGIOutput* dxgiOutput = nullptr;
    IDXGIOutput1* dxgiOutput1 = nullptr;
    IDXGIOutputDuplication* desktopDuplication = nullptr;
};

PYBIND11_MODULE(dxgi_capture, m) {
    py::class_<DXGICapture>(m, "DXGICapture")
        .def(py::init<>())
        .def("capture_frame", &DXGICapture::capture_frame);
}
