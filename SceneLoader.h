#pragma once

#include <glm/glm.hpp>
#include <embree3/rtcore.h>

#include "Scene.h"
#include "Integrator.h"

Integrator* loadScene(
    const std::string& filePath,
    RTCDevice device,
    Scene** scene);
