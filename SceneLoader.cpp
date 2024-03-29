#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <stdexcept>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Constants.h"
#include "Scene.h"
#include "Integrator.h"

#include "SceneLoader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct types {
    bool direct=false;
    bool pathtracer=false; 
    bool analyticdirect=false;
};

class SceneLoader {

private:

    RTCDevice _embreeDevice;

    glm::uvec2 _imageSize = glm::uvec2(1280, 720);
    int _maxDepth = 5;
    std::string _outputFileName = "out.png";
    glm::vec3 _cameraOrigin = glm::vec3(-1.0f, 0.0f, 0.0f);
    glm::vec3 _cameraLookAt = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 _cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    float _cameraFieldOfView = 45.0f;
    types _type;
    bool _lightstratify = false; 
    bool _nee = false;
    bool _mis = false; 
    bool _rr = false; 
    bool _cosine = false;
    bool _hemisphere = true; 
    bool _brdf = false;
    float _gamma = 1; 
    int _lightsamples = 1; 
    int _samplePerPixel = 1;
    std::vector<float> _summedArea; 
    std::vector<glm::mat4> _sphereTransforms;
    std::vector<material_t> _sphereMaterials;
    std::vector<glm::vec3> _rawVertices;
    std::vector<glm::uvec3> _indices;
    std::vector<glm::vec3> _vertices;
    std::vector<glm::vec3> _intensity;
    std::vector<material_t> _triMaterials;
    glm::mat4 curTransform = glm::mat4(1.0f);
    std::vector<glm::mat4> _transformStack;
    std::vector<quadLight_t> _quadLights;
    std::vector<directionalLight_t> _directionalLights;
    std::vector<pointLight_t> _pointLights;
    std::vector<unsigned char> _image; 
    std::vector<float> _imagef; 
    int _n = 0;
    int _width; 
    int _height; 
    glm::vec3 _curAttenuation = glm::vec3(1.0f, 0.0f, 0.0f);
    material_t _curMaterial = {
        glm::vec3(0.0f),  // diffuse
        glm::vec3(0.0f),  // specular
        1.0f,  // shininess
        glm::vec3(0.0f),  // emission
        glm::vec3(0.2f, 0.2f, 0.2f),  // ambient
        1.0f,  // roughness
        false,
        false  // mirror  
    };

public:

    SceneLoader(RTCDevice embreeDevice);
    glm::vec3 loadVec3(const std::vector<std::string>& arguments, size_t startIndex = 0);
    glm::uvec3 loadUVec3(const std::vector<std::string>& arguments, size_t startIndex = 0);
    void executeCommand(const std::string& command, const std::vector<std::string>& arguments);
    void loadSceneData(const std::string& filePath);
    Integrator* createIntegrator();
    void loadEmbreeTriangles(RTCScene embreeScene);
    void loadEmbreeSpheres(RTCScene embreeScene);
    RTCScene createEmbreeScene();
    Scene* commitSceneData();

};

SceneLoader::SceneLoader(RTCDevice embreeDevice)
    : _embreeDevice(embreeDevice)
{
}

glm::vec3 SceneLoader::loadVec3(const std::vector<std::string>& arguments, size_t startIndex)
{
    return glm::vec3(
        std::stof(arguments[startIndex]),
        std::stof(arguments[startIndex + 1]),
        std::stof(arguments[startIndex + 2]));
}

glm::uvec3 SceneLoader::loadUVec3(const std::vector<std::string>& arguments, size_t startIndex)
{
    return glm::uvec3(
        std::stoi(arguments[startIndex]),
        std::stoi(arguments[startIndex + 1]),
        std::stoi(arguments[startIndex + 2]));
}

void SceneLoader::executeCommand(
    const std::string& command,
    const std::vector<std::string>& arguments)
{
    if (command == "size") {

        _imageSize = glm::uvec2(std::stoi(arguments[0]), std::stoi(arguments[1]));

    } else if (command == "maxdepth") {

        _maxDepth = std::stoi(arguments[0]);
        if (_maxDepth == -1) _maxDepth = std::numeric_limits<int>::max();

    } else if (command == "output") {

        _outputFileName = arguments[0];

    } else if (command == "camera") {

        _cameraOrigin = loadVec3(arguments, 0);
        _cameraLookAt = loadVec3(arguments, 3);
        _cameraUp = loadVec3(arguments, 6);
        _cameraFieldOfView = std::stof(arguments[9]);

    } else if (command == "sphere") {

        glm::vec3 center = loadVec3(arguments, 0);
        float radius = std::stof(arguments[3]);

        glm::mat4 transform = glm::mat4(1.0f);
        transform = curTransform * transform;
        transform = glm::translate(transform, center);
        transform = glm::scale(transform, glm::vec3(radius));

        _sphereTransforms.push_back(transform);

        //std::cout <<  "sphere: " << _curMaterial.diffuse.z << '\n';

        _sphereMaterials.push_back(_curMaterial);

    } else if (command == "maxverts") {

        // ignore since we are using std::vector

    } else if (command == "vertex") {

        _rawVertices.push_back(loadVec3(arguments));

    } else if (command == "tri") {

        glm::uvec3 rawIndices = loadUVec3(arguments);

        _indices.push_back(glm::uvec3(
            _vertices.size(),
            _vertices.size() + 1,
            _vertices.size() + 2));

        _vertices.push_back(glm::vec3(curTransform * glm::vec4(_rawVertices[rawIndices.x], 1.0f)));
        _vertices.push_back(glm::vec3(curTransform * glm::vec4(_rawVertices[rawIndices.y], 1.0f)));
        _vertices.push_back(glm::vec3(curTransform * glm::vec4(_rawVertices[rawIndices.z], 1.0f)));

        //std::cout <<  "tri: " << _curMaterial.diffuse.z << '\n';
        _triMaterials.push_back(_curMaterial);

    } else if (command == "translate") {

        glm::vec3 translation = loadVec3(arguments);
        curTransform = glm::translate(curTransform, translation);

    } else if (command == "rotate") {

        glm::vec3 axis = loadVec3(arguments, 0);
        float radians = std::stof(arguments[3]) * PI / 180.0f;
        curTransform = glm::rotate(curTransform, radians, axis);

    } else if (command == "scale") {

        glm::vec3 scale = loadVec3(arguments);
        curTransform = glm::scale(curTransform, scale);

    } else if (command == "pushTransform") {

        _transformStack.push_back(curTransform);

    } else if (command == "popTransform") {

        curTransform = _transformStack.back();
        _transformStack.pop_back();

    } else if (command == "directional") {

        directionalLight_t light;
        light.toLight = glm::normalize(loadVec3(arguments, 0));
        light.brightness = loadVec3(arguments, 3);

        _directionalLights.push_back(light);

    } else if (command == "point") {

        pointLight_t light;
        light.point = loadVec3(arguments, 0);
        light.brightness = loadVec3(arguments, 3);
        light.attenuation = _curAttenuation;

        _pointLights.push_back(light);

    } else if (command == "attenuation") {

        _curAttenuation = loadVec3(arguments);

    } else if (command == "ambient") {

        _curMaterial.ambient = loadVec3(arguments);

    } else if (command == "diffuse") {

        _curMaterial.diffuse = loadVec3(arguments);

    } else if (command == "specular") {

        _curMaterial.specular = loadVec3(arguments);

    } else if (command == "shininess") {

        _curMaterial.shininess = std::stof(arguments[0]);

    } else if (command == "emission") {

        _curMaterial.emission = loadVec3(arguments);
    } else if (command == "integrator"){
        if(arguments[0] == "analyticdirect"){
            _type.analyticdirect = true; 
        } else if(arguments[0] == "direct"){
            _type.direct = true; 
        } else if (arguments[0] == "pathtracer"){
            _type.pathtracer = true;
        }

    } else if (command == "quadLight"){
        quadLight_t light;

        // a, b, d, c vertices insert
        light.vertices.insert(light.vertices.end(), loadVec3(arguments,0));
        light.vertices.insert(light.vertices.end(), light.vertices[0] + loadVec3(arguments, 3));
        light.vertices.insert(light.vertices.end(), light.vertices[1] + loadVec3(arguments, 6));  // a + loadVec3(arguments, 3) + loadVec3(arguments, 6);
        light.vertices.insert(light.vertices.end(), light.vertices[0] + loadVec3(arguments, 6));
               
        light.intensity = loadVec3(arguments,9);

        glm::vec3 vector = glm::cross(loadVec3(arguments, 3),loadVec3(arguments,6));
        light.area = glm::length(vector);

         _quadLights.push_back(light);

        _curMaterial.emission = loadVec3(arguments,9);

        // add in the 1st triangle
        _indices.push_back(glm::uvec3(_vertices.size(), _vertices.size() + 1, _vertices.size() + 2));

        _vertices.push_back(light.vertices[0]);
        _vertices.push_back(light.vertices[1]);
        _vertices.push_back(light.vertices[3]);

        _triMaterials.push_back(_curMaterial);

        _curMaterial.emission = loadVec3(arguments,9);

        // add in the 2nd triangle
        _indices.push_back(glm::uvec3(_vertices.size(), _vertices.size() + 1, _vertices.size() + 2));

        _vertices.push_back(light.vertices[1]);
        _vertices.push_back(light.vertices[2]);
        _vertices.push_back(light.vertices[3]);

        _triMaterials.push_back(_curMaterial);

        _curMaterial.emission = glm::vec3(0);
    } else if (command == "lightsamples") {
        _lightsamples = std::stoi(arguments[0]);
        
    } else if (command == "lightstratify"){
        if(arguments[0] == "on"){
            _lightstratify = true; 
        }
    } else if(command == "spp"){
        _samplePerPixel = std::stoi(arguments[0]);
    } else if (command == "nexteventestimation"){
        if(arguments[0] == "on"){
            _nee = true; 
        }
        else if(arguments[0] == "mis"){
            _mis = true;
        }
    } else if (command == "russianroulette"){
        if(arguments[0] == "on"){
            _rr = true;
             _maxDepth = -1; 
        }
    } else if (command == "importancesampling"){
        if(arguments[0] == "cosine"){
            _cosine = true;
            _hemisphere = false;  
        }
        else if (arguments[0] == "brdf"){
            _brdf = true; 
            _hemisphere = false; 
        }
    }else if(command == "brdf"){
        if(arguments[0] == "ggx"){
            _curMaterial.ggx = true; 
        }
    }else if(command == "roughness"){
        _curMaterial.roughness =  std::stof(arguments[0]);
        
    }else if(command == "gamma"){
        _gamma = (float) std::stoi(arguments[0]);
    }
    else if(command == "evmap"){
        // get the rgb value of the filename in arguments[0];
        // int n; // Gets if it is RGB (3) or RGBA (4)
        std::string filename = arguments[0];
        unsigned char* data = stbi_load(filename.c_str(), &_width, &_height, &_n, 4);
        float* dataf = stbi_loadf(filename.c_str(), &_width, &_height, &_n, 4);
        _n = 4;

        if (data != nullptr){
            _image = std::vector<unsigned char>(data, data + _width * _height * _n);
        }
        if (dataf != nullptr){
            _imagef = std::vector<float>(dataf, dataf + _width * _height * _n);
        }
        stbi_image_free(data);
        if(data == nullptr){
            std::cout << "Error loading image\n";
            return;
        }

        for(int y = 0; y <  _height; y++){
            for(int x = 0; x < _width; x++){
                int index = 4 * (y * _width + x);
                int  top_left = (y-1) * _width + (x-1);
                int  left  = y * _width + (x-1);
                int  top = (y-1) * _width + x;
                float top_left_n; 
                float left_n;
                float top_n; 
                if(top_left < 0){
                    //std::cout << "top_left index is oout of bounds" << '\n';
                    top_left_n = 0; 
                }
                else{
                    top_left_n = _summedArea[top_left];
                }
                if(left < 0){
                    //std::cout << "left index is oout of bounds" << '\n';
                    left_n = 0;
                }
                else{
                    left_n = _summedArea[left];
                }
                if(top < 0){
                    //std::cout << "top index is oout of bounds" << '\n';
                    top_n = 0; 
                }
                else{
                    top_n = _summedArea[top];
                }
                _summedArea.push_back( _image[index + 3] + left_n + top_n - top_left_n ); 
            }
        }

        for(int y = 0; y <  _height; y++){
            for(int x = 0; x < _width; x++){
                // get cosine of elevation of the pixel's uv coordinates
                float elevation = (1.0f - (y / _height)) * PI - PI / 2;
                int index = (y * _width + x);
                _summedArea[index] = _summedArea[index] * elevation;
            }
        }
        
        // int x = 2000;
        // int y = 500;
        // size_t index = _n * (y * _width + x);
        // std::cout << "RGB pixel @ (x=1, y=1): " 
        //       << _n << " "
        //       << static_cast<float>(_image[index + 0]) << " "
        //       << static_cast<float>(_image[index + 1]) << " "
        //       << static_cast<float>(_image[index + 2]) << " "
        //       << static_cast<float>(_image[index + 3]) << '\n';
    }
    else if(command == "mirror"){
        if(arguments[0] == "on"){
            _curMaterial.mirror = true; 
        }
        else{
            _curMaterial.mirror = false; 
        }
    }
    else {

        std::cerr << "Unknown command in scene file: '" << command << "'" << std::endl;

    }
}

void SceneLoader::loadSceneData(const std::string& filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open()) throw std::runtime_error("Could not open file: '" + filePath + "'");

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream tokenStream(line);

        std::string command;
        tokenStream >> command;

        if (command.size() == 0 || command[0] == '#') continue;

        std::vector<std::string> arguments;
        std::string argument;
        while (tokenStream >> argument) {
            arguments.push_back(argument);
        }

        executeCommand(command, arguments);
    }
}

Integrator* SceneLoader::createIntegrator()
{
    Integrator* ret;
    if( _type.direct ) {
        ret = new DirectIntegrator();
    }
    else if( _type.analyticdirect ) {
        ret = new AnalyticDirectIntegrator();
    }
    else if(_type.pathtracer) {
        ret = new PathTracerIntegrator();
    }
    else {
        ret = new RayTracerIntegrator();
    }
    return ret;
}

void SceneLoader::loadEmbreeTriangles(RTCScene embreeScene)
{
    RTCGeometry embreeTriangles = rtcNewGeometry(_embreeDevice, RTC_GEOMETRY_TYPE_TRIANGLE);

    glm::vec3* embreeVertices = reinterpret_cast<glm::vec3*>(rtcSetNewGeometryBuffer(
        embreeTriangles,
        RTC_BUFFER_TYPE_VERTEX,
        0,
        RTC_FORMAT_FLOAT3,
        sizeof(glm::vec3),
        _vertices.size()));
    std::memcpy(embreeVertices, _vertices.data(), _vertices.size() * sizeof(glm::vec3));

    glm::uvec3* embreeIndices = reinterpret_cast<glm::uvec3*>(rtcSetNewGeometryBuffer(
        embreeTriangles,
        RTC_BUFFER_TYPE_INDEX,
        0,
        RTC_FORMAT_UINT3,
        sizeof(glm::uvec3),
        _indices.size()));
    std::memcpy(embreeIndices, _indices.data(), _indices.size() * sizeof(glm::uvec3));

    rtcCommitGeometry(embreeTriangles);
    rtcAttachGeometry(embreeScene, embreeTriangles);
    rtcReleaseGeometry(embreeTriangles);
}

void SceneLoader::loadEmbreeSpheres(RTCScene embreeScene)
{
    RTCScene embreeSphereScene = rtcNewScene(_embreeDevice);

    RTCGeometry embreeSphere = rtcNewGeometry(_embreeDevice, RTC_GEOMETRY_TYPE_SPHERE_POINT);

    glm::vec4* embreeSpherePoint = reinterpret_cast<glm::vec4*>(rtcSetNewGeometryBuffer(
        embreeSphere,
        RTC_BUFFER_TYPE_VERTEX,
        0,
        RTC_FORMAT_FLOAT4,
        sizeof(glm::vec4),
        1));
    *embreeSpherePoint = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    rtcCommitGeometry(embreeSphere);
    rtcAttachGeometry(embreeSphereScene, embreeSphere);
    rtcReleaseGeometry(embreeSphere);
    rtcCommitScene(embreeSphereScene);

    for (glm::mat4 transform : _sphereTransforms) {
        RTCGeometry embreeSphereInstance = rtcNewGeometry(_embreeDevice, RTC_GEOMETRY_TYPE_INSTANCE);
        rtcSetGeometryInstancedScene(embreeSphereInstance, embreeSphereScene);
        rtcSetGeometryTimeStepCount(embreeSphereInstance, 1);
        rtcSetGeometryTransform(
            embreeSphereInstance,
            0,
            RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR,
            glm::value_ptr(transform));
        rtcCommitGeometry(embreeSphereInstance);
        rtcAttachGeometry(embreeScene, embreeSphereInstance);
        rtcReleaseGeometry(embreeSphereInstance);
    }

    rtcReleaseScene(embreeSphereScene);
}

RTCScene SceneLoader::createEmbreeScene()
{
    RTCScene embreeScene = rtcNewScene(_embreeDevice);
    loadEmbreeTriangles(embreeScene);
    loadEmbreeSpheres(embreeScene);
    rtcCommitScene(embreeScene);
    return embreeScene;
}

Scene* SceneLoader::commitSceneData()
{
    float aspectRatio = static_cast<float>(_imageSize.x) / _imageSize.y;
    glm::vec3 cameraLook = glm::normalize(_cameraLookAt - _cameraOrigin);
    glm::vec3 imagePlaneRight = glm::normalize(glm::cross(cameraLook, _cameraUp));
    glm::vec3 imagePlaneUp = glm::normalize(glm::cross(imagePlaneRight, cameraLook));

    camera_t camera;
    camera.origin = _cameraOrigin;
    camera.imagePlaneTopLeft =
        _cameraOrigin
        + cameraLook / std::tan(PI * _cameraFieldOfView / 360.0f)
        + imagePlaneUp
        - aspectRatio * imagePlaneRight;
    camera.pixelRight = (2.0f * aspectRatio / _imageSize.x) * imagePlaneRight;
    camera.pixelDown = (-2.0f / _imageSize.y) * imagePlaneUp;

    std::vector<glm::mat3> sphereNormalTransforms;
    for (size_t i = 0; i < _sphereTransforms.size(); i++) {
        sphereNormalTransforms.push_back(glm::inverseTranspose(glm::mat3(_sphereTransforms[i])));
    }

    Scene* scene = new Scene();
    scene->imageSize = _imageSize;
    scene->maxDepth = _maxDepth;
    scene->outputFileName = _outputFileName;
    scene->camera = camera;
    scene->sphereNormalTransforms = std::move(sphereNormalTransforms);
    scene->sphereMaterials = std::move(_sphereMaterials);
    scene->triMaterials = std::move(_triMaterials);
    scene->quadLights = std::move(_quadLights);
    scene->directionalLights = std::move(_directionalLights);
    scene->pointLights = std::move(_pointLights);
    scene->lightSamples = std::move(_lightsamples);
    scene->samplePerPixel = std::move(_samplePerPixel);
    scene->lightStratify = std::move(_lightstratify);
    scene->nee = std::move(_nee);
    scene->rr = std::move(_rr);
    scene->cosine = std::move(_cosine);
    scene->hemisphere = std::move(_hemisphere);
    scene->brdf = std::move(_brdf);
    scene->gamma = std::move(_gamma);
    scene->mis = std::move(_mis);
    scene->image = std::move(_image);
    scene->imagef = std::move(_imagef);
    scene->width = std::move(_width);
    scene->height = std::move(_height);
    scene->summedArea = std::move(_summedArea);
    scene->embreeScene = createEmbreeScene();
    scene->n = std::move(_n);

    return scene;
}

Integrator* loadScene(
    const std::string& filePath,
    RTCDevice embreeDevice,
    Scene** scene)
{
    SceneLoader sceneLoader(embreeDevice);
    sceneLoader.loadSceneData(filePath);
    *scene = sceneLoader.commitSceneData();
    return sceneLoader.createIntegrator();
}
