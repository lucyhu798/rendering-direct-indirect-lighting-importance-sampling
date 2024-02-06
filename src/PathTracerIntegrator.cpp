#include <iostream>
#include <glm/glm.hpp>

#include "Constants.h"
#include "Integrator.h"



inline glm::vec3 PathTracerIntegrator::alignZ(glm::vec3 v, glm::vec3 normal) {
    glm::vec3 tangent = glm::normalize(glm::cross(normal, glm::vec3(1, 2, 3)));
    glm::vec3 bitangent = glm::normalize(glm::cross(tangent, normal));
    return v.x * tangent + v.y * bitangent + v.z * normal;
}

float PathTracerIntegrator::computeG(glm::vec3 v, glm::vec3 n, float roughness) {
    float dotCheck = glm::dot(v, n);
    float angle = glm::tan(glm::acos(dotCheck));
    if ( dotCheck > 0 ) {
        return 2.0f / (1 + std::sqrt(1 + glm::pow(roughness,2) * glm::pow(angle, 2)));
    }
    else {
        return 0;
    }
}
glm::vec3 PathTracerIntegrator::computeBRDF(const material_t& material, glm::vec3 wi, glm::vec3 wo, glm::vec3 n)
{
    if (glm::dot(wi, n) <= 0.0f || glm::dot(wo, n) <= 0.0f) return glm::vec3(0.0f);

    if(material.ggx){
        glm::vec3 halfvec = glm::normalize(wi + wo); 
        float theta_h = glm::acos(glm::clamp(glm::dot(halfvec, n),0.0f,1.0f));
        glm::vec3 f_x = material.specular + ((1.0f - material.specular) * (float) glm::pow(1 - glm::dot(wi,halfvec),5));
        float g_x = computeG(wi, n, material.roughness) * computeG(wo, n, material.roughness);
        float d_x =  glm::pow(material.roughness, 2) / (PI * glm::pow(glm::cos(theta_h), 4) * glm::pow(glm::pow(material.roughness, 2)+ glm::pow(glm::tan(theta_h),2),2));

        return (material.diffuse/PI) + (f_x * g_x * d_x / (4 * glm::dot(wi, n) * glm::dot(wo, n)));
    }
    else{
        glm::vec3 r = glm::reflect(-wo, n);
        glm::vec3 diffuseLobe = material.diffuse * INV_PI;
        glm::vec3 specularLobe =
            material.specular
            * (material.shininess + 2) * INV_TWO_PI
            * std::pow(std::max(0.0f, glm::dot(r, wi)), material.shininess);

        return diffuseLobe + specularLobe;
    }
}

float PathTracerIntegrator::pdf_nee(const material_t& hitMaterial, glm::vec3 wi, glm::vec3 wo, glm::vec3 n, glm::vec3 position){
    float sum = 0.0f;
    for (const quadLight_t& light : _scene->quadLights) {
        glm::vec3 nl = glm::normalize(glm::cross(light.vertices[1] - light.vertices[0], light.vertices[3] - light.vertices[0]));
        glm::vec2 random = glm::vec2((float) rand() / ((RAND_MAX + 1u)), (float)rand() / ((RAND_MAX + 1u)));
        glm::vec3 sample = light.vertices[0] + random.x * (light.vertices[1] - light.vertices[0]) + random.y * (light.vertices[3] - light.vertices[0]);
        float R = glm::length(sample - position);
        glm::vec3 hitPosition;
        glm::vec3 hitNormal;
        material_t material;
        bool hit = _scene->castRay(position, wi, &hitPosition, &hitNormal, &material);
        if(hit && (material.emission.x > 0 || material.emission.y > 0 || material.emission.z > 0)){
            sum += glm::pow(R, 2)/ (light.area * std::abs(glm::dot(nl, wi))); 
        }
    }
    return ((float)1/_scene->quadLights.size()) * sum; 
}

float PathTracerIntegrator::pdf_brdf(const material_t& hitMaterial, glm::vec3 wi, glm::vec3 wo, glm::vec3 n){
    float specular = (hitMaterial.specular.x + hitMaterial.specular.y + hitMaterial.specular.z)/3.0f;
    float diffuse = (hitMaterial.diffuse.x + hitMaterial.diffuse.y + hitMaterial.diffuse.z)/3.0f;
    float t; 
    glm::vec3 r = glm::reflect(-wo, n);
    if(hitMaterial.ggx){
        if((specular == 0) && (diffuse == 0)){
            t = 1; 
        }
        else{
            t = std::max( 0.25f , specular / (diffuse + specular));
        } 
    }
    else{
        t = specular / (diffuse + specular); 
    }
    float p1 = (1.0f - t) * glm::dot(n, wi)/PI ; 
    float p2; 
    if(hitMaterial.ggx){
        glm::vec3 h = glm::normalize(wi + wo); 
        float theta_h = glm::acos(glm::clamp(glm::dot(h, n),0.0f,1.0f));
        float d_x =  glm::pow(hitMaterial.roughness, 2) / (PI * glm::pow(glm::cos(theta_h), 4) * glm::pow(glm::pow(hitMaterial.roughness, 2)+ glm::pow(glm::tan(theta_h),2),2));
        p2 = t * d_x * glm::dot(n, h) /(4 * glm::dot(h, wi));
    }
    else{
        p2 = t * ((hitMaterial.shininess + 1)/TWO_PI) *  std::pow(std::max(0.0f, glm::dot(r, wi)), hitMaterial.shininess);
    }
    float pdf = p1+p2; 

    return pdf;
}



glm::vec3 PathTracerIntegrator::sampleQuadLight(
    glm::vec3 direction,
    glm::vec3 hitPosition,
    glm::vec3 hitNormal,
    const material_t& hitMaterial,
    const quadLight_t& light,
    glm::vec3 sampleDirection)
{
    glm::vec3 reflectance = glm::vec3(0.0f);
    float weight = 1.0f; 
    glm::vec3 toLightDirection;

    for (int i = 0; i < _scene->lightSamples; i++) {
        int M = std::sqrt(_scene->lightSamples);
        int x = i % M;
        int y = (i /M) % M;

        glm::vec2 random = glm::vec2((float) rand() / ((RAND_MAX + 1u)), (float)rand() / ((RAND_MAX + 1u)));
        glm::vec3 lightTarget =
            light.vertices[0]
            + (x + random.x) / M * (light.vertices[1] - light.vertices[0])
            + (y + random.y) /M * (light.vertices[3] - light.vertices[0]);
        glm::vec3 toLight = lightTarget - hitPosition;
        float lightDistance = glm::length(toLight);
        toLightDirection = toLight / lightDistance;
        glm::vec3 lightNormal = glm::normalize(glm::cross(light.vertices[1] - light.vertices[0], light.vertices[3] - light.vertices[0]));

        float cosThetaO = glm::dot(lightNormal, toLightDirection);
        float cosThetaI = glm::dot(hitNormal, toLightDirection);
        if (cosThetaO < 0.0f || cosThetaI < 0.0f) continue;

        bool occluded = _scene->castOcclusionRay(hitPosition, toLightDirection, lightDistance);
        if (occluded) continue;

        reflectance += computeBRDF(hitMaterial, toLightDirection, -direction, hitNormal)
            * (cosThetaI * cosThetaO) / (lightDistance * lightDistance);
    }

    float lightArea = glm::length(glm::cross(light.vertices[1] - light.vertices[0], light.vertices[3] - light.vertices[0]));
    if(_scene->mis){
        float nee = pdf_nee(hitMaterial, toLightDirection, -direction, hitNormal, hitPosition);
        float brdf = pdf_brdf(hitMaterial, sampleDirection, -direction, hitNormal);
        weight = glm::pow(nee,2) / (glm::pow(brdf,2) + glm::pow(nee, 2)); 
    }
    return light.intensity * (lightArea / _scene->lightSamples) * reflectance * weight;
}

glm::vec3 PathTracerIntegrator::traceRay(
    glm::vec3 origin,
    glm::vec3 direction,
    int depth,
    glm::vec3 throughput)
{
    glm::vec3 outputColor = glm::vec3(0.0f, 0.0f, 0.0f);

    glm::vec3 hitPosition;
    glm::vec3 hitNormal;
    material_t hitMaterial;
    bool hit = _scene->castRay(origin, direction, &hitPosition, &hitNormal, &hitMaterial);
    if (hit) {
        if(hitMaterial.mirror == true){
            /*
            For each primary ray
                Create the root node at (0, max(0, yn − h/2)) and (w, min(h, yn + h/2)).
                Split the root at (xn , yn), the  and ((xn + w/2)%w, yn ).
                Split leaves at BRDF peaks (and neighbors as needed).
                For i = 1 to # of samples per primary ray
                    Find leaf node, R, that maximizes psplit (eq. 3).
                    Split R along the axis specified by equation 4.
                    Split neighbors of R as needed (Fig. 5).
                Calculate sampling weights for all regions (eq. 5 6 7)
            */
            EVPartitionNode* top_left = new EVPartitionNode( glm::vec2(0, 0), glm::vec2(hitNormal.x, hitNormal.y));  
            EVPartitionNode* top_middle = new EVPartitionNode( glm::vec2(hitNormal.x, 0), glm::vec2((int)(hitNormal.x + (_scene->width/2))%_scene->width,hitNormal.y)); 
            EVPartitionNode* top_right = new EVPartitionNode( glm::vec2((int)(hitNormal.x + (_scene->width/2))%_scene->width, 0), glm::vec2(_scene->width, hitNormal.y)); 
            EVPartitionNode* bottom_left = new EVPartitionNode(glm::vec2(0, hitNormal.y),glm::vec2(hitNormal.x, _scene->height)); 
            EVPartitionNode* bottom_middle = new EVPartitionNode(glm::vec2(hitNormal.x, hitNormal.y), glm::vec2((int)(hitNormal.x + (_scene->width/2))%_scene->width, _scene->height)); 
            EVPartitionNode* bottom_right = new EVPartitionNode(glm::vec2((int)(hitNormal.x + (_scene->width/2)) % _scene->width, hitNormal.y),glm::vec2(_scene->width, _scene->height));
            top_left->weight = _scene->getSummedArea( glm::vec2(0, 0), glm::vec2(hitNormal.x, hitNormal.y))/4.0f ;
            top_middle->weight = _scene->getSummedArea(glm::vec2(hitNormal.x, 0), glm::vec2((int)(hitNormal.x + (_scene->width/2))%_scene->width,hitNormal.y))/4.0f;
            top_right->weight = _scene->getSummedArea(glm::vec2((int)(hitNormal.x + (_scene->width/2))%_scene->width, 0), glm::vec2(_scene->width, hitNormal.y))/4.0f;
            bottom_left->weight = _scene->getSummedArea(glm::vec2(0, hitNormal.y),glm::vec2(hitNormal.x, _scene->height))/4.0f;
            bottom_middle->weight = _scene->getSummedArea(glm::vec2(hitNormal.x, hitNormal.y), glm::vec2((int)(hitNormal.x + (_scene->width/2))%_scene->width, _scene->height))/4.0f;
            bottom_right->weight = _scene->getSummedArea(glm::vec2((int)(hitNormal.x + (_scene->width/2)) % _scene->width, hitNormal.y),glm::vec2(_scene->width, _scene->height))/4.0f;


            // First step: Render the reflection map of environment map onto the sphere
            glm::vec3 reflection = glm::reflect(direction, hitNormal);
            // splitting leaves
            std::vector<EVPartitionNode*> nodes = {top_left, top_middle, top_right, bottom_left, bottom_middle, bottom_right};
            for ( EVPartitionNode* node : nodes ) {
                if(node->topLeft.x <= reflection.x && node->topLeft.y <= reflection.y){
                    if(node->bottomRight.x >= reflection.x && node->bottomRight.y >= reflection.y){
                        EVPartitionNode* one = new EVPartitionNode( node->topLeft, glm::vec2(reflection.x, reflection.y)); 
                        EVPartitionNode* two = new EVPartitionNode( glm::vec2(reflection.x, node->topLeft.y), glm::vec2(node->bottomRight.x, reflection.y)); 
                        EVPartitionNode* three = new EVPartitionNode(glm::vec2(node->topLeft.x, reflection.y), glm::vec2(reflection.x, node->bottomRight.y)); 
                        EVPartitionNode* four = new EVPartitionNode( glm::vec2(reflection.x, reflection.y), node->bottomRight); 
                        node->topLeftChild = one;
                        node->topRightChild = two; 
                        node->bottomLeftChild = three; 
                        node->bottomRightChild = four; 
                    }
                }
            }

            // Summed area calculation
            // _scene->getSummedArea()

            int root_weight = top_left->weight + top_middle->weight + top_right->weight + bottom_left->weight + bottom_middle->weight + bottom_right->weight; 
            int u = (_scene->width) * (atan2(reflection.x,reflection.z) + PI)/(2 * PI);
            int v = (_scene->height) * (1.0f - ( glm::asin(reflection.y) + PI / 2) /PI); 
            size_t index = 4 * (v * _scene->width + u);
            outputColor = glm::vec3((float)_scene->image[index + 0]/255.0f, (float)_scene->image[index + 1]/255.0f, (float)_scene->image[index + 2]/255.0f);
            // outputColor = glm::vec3((float)_scene->imagef[index + 0], (float)_scene->image[index + 1], (float)_scene->image[index + 2]);

            // split into two based on the normal and the width 
            // Second step: Partition the environment based on BRDF peaks
            // These peaks are the reflected rays
            

        }
        else{
            if (hitMaterial.emission.x > 0 ||hitMaterial.emission.y > 0 || hitMaterial.emission.z > 0 ) {
                if (_scene->nee == false || depth == 0) {
                    if (glm::dot(hitNormal, direction) > 0.0f) {
                        outputColor += throughput * hitMaterial.emission;
                    }
                }
            }
            else {

                if (glm::dot(hitNormal, direction) > 0.0f) hitNormal = -hitNormal;

                bool traceIndirectRay = false;
                if (_scene->nee == true ) {
                    if (depth < _scene->maxDepth) {
                        for (const quadLight_t& light : _scene->quadLights) {
                            outputColor += throughput * sampleQuadLight(
                                direction,
                                hitPosition,
                                hitNormal,
                                hitMaterial,
                                light,
                                glm::vec3(0.0f));
                        }
                    }
                    traceIndirectRay = (depth < _scene->maxDepth - 1);
                }
                else if(_scene->mis == true){

                    glm::vec3 nee = glm::vec3(0),
                              brdf = nee;

                    // BRDF Calculation
                    glm::vec3 samp = glm::vec3((float) rand() / ((RAND_MAX + 1u)), (float)rand() / ((RAND_MAX + 1u)),(float)rand() / ((RAND_MAX + 1u)));
                    float phi;
                    float theta ;
                    float cosTheta;
                    float sinPhi;
                    float cosPhi;
                    float sinTheta;
                    glm::vec3 sampleDirection;
                    float specular = (hitMaterial.specular.x + hitMaterial.specular.y + hitMaterial.specular.z)/3.0f;
                    float diffuse = (hitMaterial.diffuse.x + hitMaterial.diffuse.y + hitMaterial.diffuse.z)/3.0f;
                    float t; 
                    if(hitMaterial.ggx){
                        if((specular == 0) && (diffuse == 0)){
                            t = 1; 
                        }
                        else{
                            t = std::max( 0.25f , specular / (diffuse + specular));
                        } 
                    }
                    else{
                    t = specular / (diffuse + specular); 
                    }
                    phi = TWO_PI * samp.z;
                    if(samp.x <= t){
                        if(hitMaterial.ggx){
                            theta = glm::atan((hitMaterial.roughness * std::sqrt(samp.y))/std::sqrt(1 - samp.y));
                        }
                        else{
                            theta = glm::acos(glm::clamp(glm::pow(samp.y, 1.0f/(hitMaterial.shininess + 1.0f)),0.0f,1.0f));
                        }
                    }
                    else{
                        theta = glm::acos(glm::clamp(std::sqrt(samp.y),0.0f,1.0f));
                    }
                    cosTheta = std::cos(theta);
                    sinPhi = std::sin(phi);
                    cosPhi = std::cos(phi);
                    sinTheta = std::sin(theta);
                    if(samp.x <= t){
                        if(hitMaterial.ggx){
                            sampleDirection = alignZ(glm::vec3(
                            cosPhi * sinTheta,
                            sinPhi * sinTheta,
                            cosTheta),hitNormal);
                            sampleDirection = glm::reflect(direction, sampleDirection);
                        }
                        else{
                            sampleDirection = alignZ(glm::vec3(
                            cosPhi * sinTheta,
                            sinPhi * sinTheta,
                            cosTheta), glm::reflect(direction, hitNormal));
                        }
                    }
                    else{
                        sampleDirection = alignZ(glm::vec3(
                        cosPhi * sinTheta,
                        sinPhi * sinTheta,
                        cosTheta), hitNormal);
                
                    }
                    glm::vec3 w_i = sampleDirection;
                    glm::vec3 w_o = -direction;
                    glm::vec3 n = hitNormal;

                    float pdf = pdf_brdf(hitMaterial, w_i, w_o, n); 

                    // NEE Calculation here to pass sampleDirection
                    for (const quadLight_t& light : _scene->quadLights) {
                        nee += throughput * sampleQuadLight(
                            direction,
                            hitPosition,
                            hitNormal,
                            hitMaterial,
                            light,
                            sampleDirection); 
                    }
                    // Use atan2 for deriving theta and phi
                    // theta = acos(y)
                    // phi = atan2(x,z)

                    // To get which pixel:
                    // Theta is in [0, PI)
                    // Phi is in [-pi, pi)
                    // u = theta / pi * width
                    // v = (phi + pi) / (2 pi) * height

                    // Note that given z is at the center of the environment map, 
                    //      the sign of phi determines the quadrant

                    // To render the environment map, you check what point the ray intersects with the environment map
                    // Then you return that output color

                    // If we fail, break the implementation into the steps given in the research paper
                    // One way of starting is assuming BRDF is diffuse; this is approximating NEE

                    // (L_e * f * v)/P(\omega) is the sampler

                    // For the final project, implement NEE with MIS or Two stage importance sampling

                    // Hierarchical warping can also be used to sample 
                    //     the environment map by creating leaves and weights
                    //     with a probability of w1 / (w1 + w2) choose the leaves and then sample the leaf
                    // If we want the right way, section 3 of research paper (last paragraph / appendix A) has code that can be referenced
                    // Summed area table used to compute
                    // Start with diffuse BRDF and then add glossy sampling of environment map into it
                    // We need to switch to EXR format for the environment map because the PNG doesn't have enough dynamic range

                    throughput *=
                        computeBRDF(hitMaterial, w_i, w_o, n) * glm::dot(w_i, n)/pdf;

                    float pdff_nee = pdf_nee(hitMaterial, sampleDirection, -direction, hitNormal, hitPosition);
                    float weight = glm::pow(pdf,2) / (glm::pow(pdf,2) + glm::pow(pdff_nee,2)); 
                    brdf += weight * traceRay(hitPosition, sampleDirection, depth + 1 , throughput);

                    // Current progress on MIS: 
                    // Both brdf and nee output correctly but the weights are not correct
                    // It seems to be that the weight needs to be all PDFs of each light
                    // This means creating the toLightDirection prematurely, calculating the weight,
                    // and then calculating the NEE
                    //
                    // For that, we only need one toLightDirection to use for that NEE PDF
                    // We'll pass that in as the initial PDF, though it'll be the exception
                    outputColor =  brdf + (nee/(float) _scene->quadLights.size());

                }
                else {
                    traceIndirectRay = (depth < _scene->maxDepth);
                }

                if (traceIndirectRay) {
                    if (_scene->rr) {
                        float continueProbability = std::min(std::max(std::max(throughput.x, throughput.y), throughput.z), 1.0f);
                        glm::vec2 random = glm::vec2((float) rand() / ((RAND_MAX + 1u)), (float)rand() / ((RAND_MAX + 1u)));
                        if (random.x < continueProbability) {
                            throughput /= continueProbability;
                        }
                        else {
                            traceIndirectRay = false;
                        }
                    }
                }
                if (traceIndirectRay) {
                    glm::vec3 samp = glm::vec3((float) rand() / ((RAND_MAX + 1u)), (float)rand() / ((RAND_MAX + 1u)),(float)rand() / ((RAND_MAX + 1u)));
                    float phi;
                    float theta ;
                    float cosTheta;
                    float sinPhi;
                    float cosPhi;
                    float sinTheta;
                    glm::vec3 sampleDirection;
                    float specular = (hitMaterial.specular.x + hitMaterial.specular.y + hitMaterial.specular.z)/3.0f;
                    float diffuse = (hitMaterial.diffuse.x + hitMaterial.diffuse.y + hitMaterial.diffuse.z)/3.0f;
                    float t; 
                    if(hitMaterial.ggx){
                        if((specular == 0) && (diffuse == 0)){
                            t = 1; 
                        }
                        else{
                            t = std::max( 0.25f , specular / (diffuse + specular));
                        } 
                    }
                    else{
                    t = specular / (diffuse + specular); 
                    }
                    if(_scene->cosine){
                        phi = TWO_PI * samp.x;
                        theta = glm::acos(std::sqrt(samp.y));
                        cosTheta = std::cos(theta);
                        sinPhi = std::sin(phi);
                        cosPhi = std::cos(phi);
                        sinTheta = std::sin(theta);
                        sampleDirection = alignZ(glm::vec3(
                            cosPhi * sinTheta,
                            sinPhi * sinTheta,
                            cosTheta), hitNormal);
                        throughput *=
                            computeBRDF(hitMaterial, sampleDirection, -direction, hitNormal)
                            * PI;
                    }
                    else if(_scene->brdf){
                        phi = TWO_PI * samp.z;
                        if(samp.x <= t){
                            if(hitMaterial.ggx){
                                theta = glm::atan((hitMaterial.roughness * std::sqrt(samp.y))/std::sqrt(1 - samp.y));
                            }
                            else{
                                theta = glm::acos(glm::clamp(glm::pow(samp.y, 1.0f/(hitMaterial.shininess + 1.0f)),0.0f,1.0f));
                            }
                        }
                        else{
                            theta = glm::acos(glm::clamp(std::sqrt(samp.y),0.0f,1.0f));
                        }
                        cosTheta = std::cos(theta);
                        sinPhi = std::sin(phi);
                        cosPhi = std::cos(phi);
                        sinTheta = std::sin(theta);
                        if(samp.x <= t){
                            if(hitMaterial.ggx){
                                sampleDirection = alignZ(glm::vec3(
                                cosPhi * sinTheta,
                                sinPhi * sinTheta,
                                cosTheta),hitNormal);
                                sampleDirection = glm::reflect(direction, sampleDirection);
                            }
                            else{
                                sampleDirection = alignZ(glm::vec3(
                                cosPhi * sinTheta,
                                sinPhi * sinTheta,
                                cosTheta), glm::reflect(direction, hitNormal));
                            }
                        }
                        else{
                            sampleDirection = alignZ(glm::vec3(
                            cosPhi * sinTheta,
                            sinPhi * sinTheta,
                            cosTheta), hitNormal);
                    
                        }
                        glm::vec3 w_i = sampleDirection;
                        glm::vec3 w_o = -direction;
                        glm::vec3 n = hitNormal;

                        float pdf = pdf_brdf(hitMaterial, w_i, w_o, n); 
                        throughput *=
                            computeBRDF(hitMaterial, w_i, w_o, n) * glm::dot(w_i, n)/pdf;
                    }
                    else {
                        phi = TWO_PI * samp.x;
                        cosTheta = samp.y;
                        sinPhi = std::sin(phi);
                        cosPhi = std::cos(phi);
                        sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);

                        sampleDirection = alignZ(glm::vec3(
                            cosPhi * sinTheta,
                            sinPhi * sinTheta,
                            cosTheta), hitNormal);
                        throughput *=
                            computeBRDF(hitMaterial, sampleDirection, -direction, hitNormal)
                            * glm::dot(sampleDirection, hitNormal)
                            * TWO_PI;
                    }
                    outputColor += traceRay(hitPosition, sampleDirection, depth + 1, throughput);
                }
            }
        }
    }
    else {
        // Background environment map lighting
        int u = (_scene->width) * (atan2(direction.x,direction.z) + PI)/(2 * PI);
        int v = (_scene->height) * (1.0f - ( glm::asin(direction.y) + PI / 2) /PI); 
        size_t index = 4 * (v * _scene->width + u);
        outputColor = glm::vec3((float)_scene->image[index + 0]/255.0f, (float)_scene->image[index + 1]/255.0f, (float)_scene->image[index + 2]/255.0f);
    }
    //std::cout << " not mirror: " << outputColor.x << " " << outputColor.y << " " << outputColor.z << '\n';
    return outputColor;
}

glm::vec3 PathTracerIntegrator::traceRay(glm::vec3 origin, glm::vec3 direction)
{
    return traceRay(origin, direction, 0, glm::vec3(1.0f));
}