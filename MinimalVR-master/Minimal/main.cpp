/************************************************************************************

Authors     :   Bradley Austin Davis <bdavis@saintandreas.org>
Copyright   :   Copyright Brad Davis. All Rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

************************************************************************************/


#include <iostream>
#include <memory>
#include <exception>
#include <algorithm>
#include <Windows.h>
#include <vector>
#include <string>
#include <ctime>
#include <utility>
#include "mesh.h"
#include "model.h"

#define __STDC_FORMAT_MACROS 1

#define FAIL(X) throw std::runtime_error(X)

///////////////////////////////////////////////////////////////////////////////
//
// GLM is a C++ math library meant to mirror the syntax of GLSL 
//

#include <glm/glm.hpp>
#include <glm/gtc/noise.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>

// Import the most commonly used types into the default namespace
using glm::ivec3;
using glm::ivec2;
using glm::uvec2;
using glm::mat3;
using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::quat;

mat3 rotate(const float degrees, const vec3& axis)
{
	glm::normalize(axis);
	float angle = glm::radians(degrees);
	return (float)cos(angle)*mat3(1, 0, 0, 0, 1, 0, 0, 0, 1) +
		(float)(1 - cos(angle))*mat3(
			axis.x*axis.x, axis.x*axis.y, axis.x*axis.z,
			axis.x*axis.y, axis.y*axis.y, axis.y*axis.z,
			axis.x*axis.z, axis.y*axis.z, axis.z*axis.z) +
			(float)sin(angle)*mat3(
				0, axis.z, -axis.y,
				-axis.z, 0, axis.x,
				axis.y, -axis.x, 0);
}

mat4 scale(const float &sx, const float &sy, const float &sz)
{
	mat4 ret(
		sx, 0.0, 0.0, 0.0,
		0.0, sy, 0.0, 0.0,
		0.0, 0.0, sz, 0.0,
		0.0, 0.0, 0.0, 1.0);

	return ret;
}

mat4 translate(const float &tx, const float &ty, const float &tz)
{
	mat4 ret(
		1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		tx, ty, tz, 1.0);
	return ret;
}


bool left_trig = false;
bool right_trig = false;
bool win = false;
bool lost = false;
bool reset_flag = false;
///////////////////////////////////////////////////////////////////////////////
//
// GLEW gives cross platform access to OpenGL 3.x+ functionality.  
//

#include <GL/glew.h>

bool checkFramebufferStatus(GLenum target = GL_FRAMEBUFFER) {
	GLuint status = glCheckFramebufferStatus(target);
	switch (status) {
	case GL_FRAMEBUFFER_COMPLETE:
		return true;
		break;

	case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
		std::cerr << "framebuffer incomplete attachment" << std::endl;
		break;

	case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
		std::cerr << "framebuffer missing attachment" << std::endl;
		break;

	case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
		std::cerr << "framebuffer incomplete draw buffer" << std::endl;
		break;

	case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
		std::cerr << "framebuffer incomplete read buffer" << std::endl;
		break;

	case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
		std::cerr << "framebuffer incomplete multisample" << std::endl;
		break;

	case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
		std::cerr << "framebuffer incomplete layer targets" << std::endl;
		break;

	case GL_FRAMEBUFFER_UNSUPPORTED:
		std::cerr << "framebuffer unsupported internal format or image" << std::endl;
		break;

	default:
		std::cerr << "other framebuffer error" << std::endl;
		break;
	}

	return false;
}

bool checkGlError() {
	GLenum error = glGetError();
	if (!error) {
		return false;
	}
	else {
		switch (error) {
		case GL_INVALID_ENUM:
			std::cerr << ": An unacceptable value is specified for an enumerated argument.The offending command is ignored and has no other side effect than to set the error flag.";
			break;
		case GL_INVALID_VALUE:
			std::cerr << ": A numeric argument is out of range.The offending command is ignored and has no other side effect than to set the error flag";
			break;
		case GL_INVALID_OPERATION:
			std::cerr << ": The specified operation is not allowed in the current state.The offending command is ignored and has no other side effect than to set the error flag..";
			break;
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			std::cerr << ": The framebuffer object is not complete.The offending command is ignored and has no other side effect than to set the error flag.";
			break;
		case GL_OUT_OF_MEMORY:
			std::cerr << ": There is not enough memory left to execute the command.The state of the GL is undefined, except for the state of the error flags, after this error is recorded.";
			break;
		case GL_STACK_UNDERFLOW:
			std::cerr << ": An attempt has been made to perform an operation that would cause an internal stack to underflow.";
			break;
		case GL_STACK_OVERFLOW:
			std::cerr << ": An attempt has been made to perform an operation that would cause an internal stack to overflow.";
			break;
		}
		return true;
	}
}

void glDebugCallbackHandler(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *msg, GLvoid* data) {
	OutputDebugStringA(msg);
	std::cout << "debug call: " << msg << std::endl;
}

/************************************************************************************
* GL helpers
************************************************************************************/

enum {
	VERTEX,
	FRAGMENT,
	SHADER_COUNT
};

static GLuint _compileProgramFromSource(const char vertexShaderSource[], const char fragmentShaderSource[], size_t errorBufferSize, char* errorBuffer) {
	const GLenum types[SHADER_COUNT] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
	const char* sources[SHADER_COUNT] = { vertexShaderSource, fragmentShaderSource };
	GLint shaders[SHADER_COUNT] { 0, 0 };
	bool success = true;

	// Compile all of the shader program objects
	for (int i = 0; i < SHADER_COUNT; ++i) {
		shaders[i] = glCreateShader(types[i]);
		glShaderSource(shaders[i], 1, &sources[i], NULL);
		glCompileShader(shaders[i]);

		GLint compileSuccess;
		glGetShaderiv(shaders[i], GL_COMPILE_STATUS, &compileSuccess);
		if (!compileSuccess) {
			glGetShaderInfoLog(shaders[i], (GLsizei)errorBufferSize, NULL, errorBuffer);
			success = false;
			break;
		}
	}

	// Create and link the program
	GLuint program = 0;
	if (success) {
		program = glCreateProgram();
		for (int i = 0; i < SHADER_COUNT; ++i) {
			glAttachShader(program, shaders[i]);
		}
		glLinkProgram(program);

		GLint linkSuccess;
		glGetProgramiv(program, GL_LINK_STATUS, &linkSuccess);
		if (!linkSuccess) {
			glGetProgramInfoLog(program, (GLsizei)errorBufferSize, NULL, errorBuffer);
			glDeleteProgram(program);
			program = 0;
		}
	}
	for (int i = 0; i < SHADER_COUNT; ++i) {
		if (shaders[i]) {
			glDeleteShader(shaders[i]);
		}
	}
	return program;
}

static std::string ExePath() {
	char buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");
	return std::string(buffer).substr(0, pos+1);
}

static GLuint _compileProgramFromFiles(const char vertexShaderPath[], const char fragmentShaderPath[], size_t errorBufferSize, char* errorBuffer) {
	const char* fileSources[SHADER_COUNT] = { vertexShaderPath, fragmentShaderPath };
	char* fileBuffers[SHADER_COUNT] = { NULL, NULL };
	bool success = true;

	// Load each of the shader files
	for (int i = 0; i < SHADER_COUNT; ++i) {
		std::string fullPath = ExePath();
		fullPath += fileSources[i];
		FILE* file = fopen(fullPath.c_str(), "rb");
		if (!file) {
			strncpy(errorBuffer, "Failed to open shader files.", errorBufferSize);
			success = false;
			break;
		}
		fseek(file, 0, SEEK_END);
		long offset = ftell(file);
		fseek(file, 0, SEEK_SET);
		fileBuffers[i] = (char*)malloc(offset + 1);
		fread(fileBuffers[i], 1, offset, file);
		fileBuffers[i][offset] = '\0';
	}

	// Compile the program
	GLuint program = 0;
	if (success) {
		program = _compileProgramFromSource(fileBuffers[VERTEX], fileBuffers[FRAGMENT], errorBufferSize, errorBuffer);
	}

	// Clean up the loaded data
	for (int i = 0; i < SHADER_COUNT; ++i) {
		if (fileBuffers[i]) {
			free(fileBuffers[i]);
		}
	}
	return program;
}

//////////////////////////////////////////////////////////////////////
//
// GLFW provides cross platform window creation
//

#include <GLFW/glfw3.h>

namespace glfw {
	inline GLFWwindow * createWindow(const uvec2 & size, const ivec2 & position = ivec2(INT_MIN)) {
		GLFWwindow * window = glfwCreateWindow(size.x, size.y, "glfw", nullptr, nullptr);
		if (!window) {
			FAIL("Unable to create rendering window");
		}
		if ((position.x > INT_MIN) && (position.y > INT_MIN)) {
			glfwSetWindowPos(window, position.x, position.y);
		}
		return window;
	}
}

// A class to encapsulate using GLFW to handle input and render a scene
class GlfwApp {

protected:
	uvec2 windowSize;
	ivec2 windowPosition;
	GLFWwindow * window{ nullptr };
	unsigned int frame{ 0 };

public:
	// Run the main loop

	GlfwApp() {
		// Initialize the GLFW system for creating and positioning windows
		if (!glfwInit()) {
			FAIL("Failed to initialize GLFW");
		}
		glfwSetErrorCallback(ErrorCallback);
	}

	virtual ~GlfwApp() {
		if (nullptr != window) {
			glfwDestroyWindow(window);
		}
		glfwTerminate();
	}

	virtual int run() {
		preCreate();

		window = createRenderingTarget(windowSize, windowPosition);

		if (!window) {
			std::cout << "Unable to create OpenGL window" << std::endl;
			return -1;
		}

		postCreate();

		initGl();

		while (!glfwWindowShouldClose(window)) {
			++frame;
			glfwPollEvents();
			update();
			draw();
			finishFrame();
		}

		shutdownGl();

		return 0;
	}


protected:
	virtual GLFWwindow * createRenderingTarget(uvec2 & size, ivec2 & pos) = 0;

	virtual void draw() = 0;

	void preCreate() {
		glfwWindowHint(GLFW_DEPTH_BITS, 16);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
	}


	void postCreate() {
		glfwSetWindowUserPointer(window, this);
		glfwSetKeyCallback(window, KeyCallback);
		glfwSetMouseButtonCallback(window, MouseButtonCallback);
		glfwMakeContextCurrent(window);

		// Initialize the OpenGL bindings
		// For some reason we have to set this experminetal flag to properly
		// init GLEW if we use a core context.
		glewExperimental = GL_TRUE;
		if (0 != glewInit()) {
			FAIL("Failed to initialize GLEW");
		}
		glGetError();

		if (GLEW_KHR_debug) {
			GLint v;
			glGetIntegerv(GL_CONTEXT_FLAGS, &v);
			if (v & GL_CONTEXT_FLAG_DEBUG_BIT) {
				//glDebugMessageCallback(glDebugCallbackHandler, this);
			}
		}
	}

	virtual void initGl() {
	}

	virtual void shutdownGl() {
	}

	virtual void finishFrame() {
		glfwSwapBuffers(window);
	}

	virtual void destroyWindow() {
		glfwSetKeyCallback(window, nullptr);
		glfwSetMouseButtonCallback(window, nullptr);
		glfwDestroyWindow(window);
	}

	virtual void onKey(int key, int scancode, int action, int mods) {
		if (GLFW_PRESS != action) {
			return;
		}

		switch (key) {
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, 1);
			return;
		}
	}

	virtual void update() {}

	virtual void onMouseButton(int button, int action, int mods) {}

protected:
	virtual void viewport(const ivec2 & pos, const uvec2 & size) {
		glViewport(pos.x, pos.y, size.x, size.y);
	}

private:

	static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
		GlfwApp * instance = (GlfwApp *)glfwGetWindowUserPointer(window);
		instance->onKey(key, scancode, action, mods);
	}

	static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
		GlfwApp * instance = (GlfwApp *)glfwGetWindowUserPointer(window);
		instance->onMouseButton(button, action, mods);
	}

	static void ErrorCallback(int error, const char* description) {
		FAIL(description);
	}
};

//////////////////////////////////////////////////////////////////////
//
// The Oculus VR C API provides access to information about the HMD
//

#include <OVR_CAPI.h>
#include <OVR_CAPI_GL.h>
#include <OVR_Platform.h>

#include <OVR_Avatar.h>

#include <map>
#include <chrono>

/************************************************************************************
* Constants
************************************************************************************/

#define MIRROR_SAMPLE_APP_ID "958062084316416"
#define MIRROR_WINDOW_WIDTH 800
#define MIRROR_WINDOW_HEIGHT 600

// Disable MIRROR_ALLOW_OVR to force 2D rendering
#define MIRROR_ALLOW_OVR true


/************************************************************************************
* Static state
************************************************************************************/

static GLuint _skinnedMeshProgram;
static GLuint _skinnedMeshPBSProgram;
static GLuint _debugLineProgram;
static GLuint _debugVertexArray;
static GLuint _debugVertexBuffer;
static ovrAvatar* _avatar;
static int _loadingAssets;
static float _elapsedSeconds;
static std::map<ovrAvatarAssetID, void*> _assetMap;
std::chrono::steady_clock::time_point lastTime;
static glm::vec4 laserColorLeft(0, 1, 0, 1);
static glm::vec4 laserColorRight(0, 1, 0, 1);

pair<vec3, ovrQuatf> left_line_pos;
pair<vec3, ovrQuatf> right_line_pos;

ovrSession tempOvrSession;

/************************************************************************************
* Math helpers and type conversions
************************************************************************************/

static glm::vec3 _glmFromOvrVector(const ovrVector3f& ovrVector)
{
	return glm::vec3(ovrVector.x, ovrVector.y, ovrVector.z);
}

static glm::quat _glmFromOvrQuat(const ovrQuatf& ovrQuat)
{
	return glm::quat(ovrQuat.w, ovrQuat.x, ovrQuat.y, ovrQuat.z);
}

static glm::vec3 _nextPointFromPosition(const glm::vec3& position, const glm::quat& orientation) {
	float w = orientation.w;
	float x = orientation.x;
	float y = orientation.y;
	float z = orientation.z;

	glm::vec3 result = position + glm::vec3(((2 * x * z) - (2 * y * w)), ((2 * y*z) + (2 * x * w)), (1 - (2 * pow(x, 2)) - (2 * pow(y, 2))));

	return result;
}

static void _glmFromOvrAvatarTransform(const ovrAvatarTransform& transform, glm::mat4* target) {
	glm::vec3 position(transform.position.x, transform.position.y, transform.position.z);
	glm::quat orientation(transform.orientation.w, transform.orientation.x, transform.orientation.y, transform.orientation.z);
	glm::vec3 scale(transform.scale.x, transform.scale.y, transform.scale.z);
	*target = glm::translate(position) * glm::mat4_cast(orientation) * glm::scale(scale);
}

static void _ovrAvatarTransformFromGlm(const glm::vec3& position, const glm::quat& orientation, const glm::vec3& scale, ovrAvatarTransform* target) {
	target->position.x = position.x;
	target->position.y = position.y;
	target->position.z = position.z;
	target->orientation.x = orientation.x;
	target->orientation.y = orientation.y;
	target->orientation.z = orientation.z;
	target->orientation.w = orientation.w;
	target->scale.x = scale.x;
	target->scale.y = scale.y;
	target->scale.z = scale.z;
}

static void _ovrAvatarTransformFromGlm(const glm::mat4& matrix, ovrAvatarTransform* target) {
	glm::vec3 scale;
	glm::quat orientation;
	glm::vec3 translation;
	glm::vec3 skew;
	glm::vec4 perspective;
	glm::decompose(matrix, scale, orientation, translation, skew, perspective);
	_ovrAvatarTransformFromGlm(translation, orientation, scale, target);
}

static void _ovrAvatarHandInputStateFromOvr(const ovrAvatarTransform& transform, const ovrInputState& inputState, ovrHandType hand, ovrAvatarHandInputState* state)
{
	state->transform = transform;
	state->buttonMask = 0;
	state->touchMask = 0;
	state->joystickX = inputState.Thumbstick[hand].x;
	state->joystickY = inputState.Thumbstick[hand].y;
	state->indexTrigger = inputState.IndexTrigger[hand];
	state->handTrigger = inputState.HandTrigger[hand];
	state->isActive = false;
	if (hand == ovrHand_Left)
	{
		if (inputState.Buttons & ovrButton_X) state->buttonMask |= ovrAvatarButton_One;
		if (inputState.Buttons & ovrButton_Y) state->buttonMask |= ovrAvatarButton_Two;
		if (inputState.Buttons & ovrButton_Enter) state->buttonMask |= ovrAvatarButton_Three;
		if (inputState.Buttons & ovrButton_LThumb) state->buttonMask |= ovrAvatarButton_Joystick;
		if (inputState.Touches & ovrTouch_X) state->touchMask |= ovrAvatarTouch_One;
		if (inputState.Touches & ovrTouch_Y) state->touchMask |= ovrAvatarTouch_Two;
		if (inputState.Touches & ovrTouch_LThumb) state->touchMask |= ovrAvatarTouch_Joystick;
		if (inputState.Touches & ovrTouch_LThumbRest) state->touchMask |= ovrAvatarTouch_ThumbRest;
		if (inputState.Touches & ovrTouch_LIndexTrigger) state->touchMask |= ovrAvatarTouch_Index;
		if (inputState.Touches & ovrTouch_LIndexPointing) state->touchMask |= ovrAvatarTouch_Pointing;
		if (inputState.Touches & ovrTouch_LThumbUp) state->touchMask |= ovrAvatarTouch_ThumbUp;
		state->isActive = (inputState.ControllerType & ovrControllerType_LTouch) != 0;
	}
	else if (hand == ovrHand_Right)
	{
		if (inputState.Buttons & ovrButton_A) state->buttonMask |= ovrAvatarButton_One;
		if (inputState.Buttons & ovrButton_B) state->buttonMask |= ovrAvatarButton_Two;
		if (inputState.Buttons & ovrButton_Home) state->buttonMask |= ovrAvatarButton_Three;
		if (inputState.Buttons & ovrButton_RThumb) state->buttonMask |= ovrAvatarButton_Joystick;
		if (inputState.Touches & ovrTouch_A) state->touchMask |= ovrAvatarTouch_One;
		if (inputState.Touches & ovrTouch_B) state->touchMask |= ovrAvatarTouch_Two;
		if (inputState.Touches & ovrTouch_RThumb) state->touchMask |= ovrAvatarTouch_Joystick;
		if (inputState.Touches & ovrTouch_RThumbRest) state->touchMask |= ovrAvatarTouch_ThumbRest;
		if (inputState.Touches & ovrTouch_RIndexTrigger) state->touchMask |= ovrAvatarTouch_Index;
		if (inputState.Touches & ovrTouch_RIndexPointing) state->touchMask |= ovrAvatarTouch_Pointing;
		if (inputState.Touches & ovrTouch_RThumbUp) state->touchMask |= ovrAvatarTouch_ThumbUp;
		state->isActive = (inputState.ControllerType & ovrControllerType_RTouch) != 0;
	}
}

static void _computeWorldPose(const ovrAvatarSkinnedMeshPose& localPose, glm::mat4* worldPose)
{
	for (uint32_t i = 0; i < localPose.jointCount; ++i)
	{
		glm::mat4 local;
		_glmFromOvrAvatarTransform(localPose.jointTransform[i], &local);

		int parentIndex = localPose.jointParents[i];
		if (parentIndex < 0)
		{
			worldPose[i] = local;
		}
		else
		{
			worldPose[i] = worldPose[parentIndex] * local;
		}
	}
}

static glm::mat4 _computeReflectionMatrix(const glm::vec4& plane)
{
	return glm::mat4(
		1.0f - 2.0f * plane.x * plane.x,
		-2.0f * plane.x * plane.y,
		-2.0f * plane.x * plane.z,
		-2.0f * plane.w * plane.x,

		-2.0f * plane.y * plane.x,
		1.0f - 2.0f * plane.y * plane.y,
		-2.0f * plane.y * plane.z,
		-2.0f * plane.w * plane.y,

		-2.0f * plane.z * plane.x,
		-2.0f * plane.z * plane.y,
		1.0f - 2.0f * plane.z * plane.z,
		-2.0f * plane.w * plane.z,

		0.0f,
		0.0f,
		0.0f,
		1.0f
	);
}


/************************************************************************************
* Wrappers for GL representations of avatar assets
************************************************************************************/

struct MeshData {
	GLuint vertexArray;
	GLuint vertexBuffer;
	GLuint elementBuffer;
	GLuint elementCount;
	glm::mat4 bindPose[OVR_AVATAR_MAXIMUM_JOINT_COUNT];
	glm::mat4 inverseBindPose[OVR_AVATAR_MAXIMUM_JOINT_COUNT];
};

struct TextureData {
	GLuint textureID;
};

static MeshData* _loadMesh(const ovrAvatarMeshAssetData* data)
{
	MeshData* mesh = new MeshData();

	// Create the vertex array and buffer
	glGenVertexArrays(1, &mesh->vertexArray);
	glGenBuffers(1, &mesh->vertexBuffer);
	glGenBuffers(1, &mesh->elementBuffer);

	// Bind the vertex buffer and assign the vertex data
	glBindVertexArray(mesh->vertexArray);
	glBindBuffer(GL_ARRAY_BUFFER, mesh->vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, data->vertexCount * sizeof(ovrAvatarMeshVertex), data->vertexBuffer, GL_STATIC_DRAW);

	// Bind the index buffer and assign the index data
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->elementBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, data->indexCount * sizeof(GLushort), data->indexBuffer, GL_STATIC_DRAW);
	mesh->elementCount = data->indexCount;

	// Fill in the array attributes
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ovrAvatarMeshVertex), &((ovrAvatarMeshVertex*)0)->x);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ovrAvatarMeshVertex), &((ovrAvatarMeshVertex*)0)->nx);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(ovrAvatarMeshVertex), &((ovrAvatarMeshVertex*)0)->tx);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(ovrAvatarMeshVertex), &((ovrAvatarMeshVertex*)0)->u);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(4, 4, GL_BYTE, GL_FALSE, sizeof(ovrAvatarMeshVertex), &((ovrAvatarMeshVertex*)0)->blendIndices);
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(ovrAvatarMeshVertex), &((ovrAvatarMeshVertex*)0)->blendWeights);
	glEnableVertexAttribArray(5);

	// Clean up
	glBindVertexArray(0);

	// Translate the bind pose
	_computeWorldPose(data->skinnedBindPose, mesh->bindPose);
	for (uint32_t i = 0; i < data->skinnedBindPose.jointCount; ++i)
	{
		mesh->inverseBindPose[i] = glm::inverse(mesh->bindPose[i]);
	}
	return mesh;
}

static TextureData* _loadTexture(const ovrAvatarTextureAssetData* data)
{
	// Create a texture
	TextureData* texture = new TextureData();
	glGenTextures(1, &texture->textureID);
	glBindTexture(GL_TEXTURE_2D, texture->textureID);

	// Load the image data
	switch (data->format)
	{

		// Handle uncompressed image data
	case ovrAvatarTextureFormat_RGB24:
		for (uint32_t level = 0, offset = 0, width = data->sizeX, height = data->sizeY; level < data->mipCount; ++level)
		{
			glTexImage2D(GL_TEXTURE_2D, level, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data->textureData + offset);
			offset += width * height * 3;
			width /= 2;
			height /= 2;
		}
		break;

		// Handle compressed image data
	case ovrAvatarTextureFormat_DXT1:
	case ovrAvatarTextureFormat_DXT5:
		GLenum glFormat;
		int blockSize;
		if (data->format == ovrAvatarTextureFormat_DXT1)
		{
			blockSize = 8;
			glFormat = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
		}
		else
		{
			blockSize = 16;
			glFormat = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
		}

		for (uint32_t level = 0, offset = 0, width = data->sizeX, height = data->sizeY; level < data->mipCount; ++level)
		{
			GLsizei levelSize = (width < 4 || height < 4) ? blockSize : blockSize * (width / 4) * (height / 4);
			glCompressedTexImage2D(GL_TEXTURE_2D, level, glFormat, width, height, 0, levelSize, data->textureData + offset);
			offset += levelSize;
			width /= 2;
			height /= 2;
		}
		break;
	}
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	return texture;
}


/************************************************************************************
* Rendering functions
************************************************************************************/

static void _setTextureSampler(GLuint program, int textureUnit, const char uniformName[], ovrAvatarAssetID assetID)
{
	GLuint textureID = 0;
	if (assetID)
	{
		void* data = _assetMap[assetID];
		TextureData* textureData = (TextureData*)data;
		textureID = textureData->textureID;
	}
	glActiveTexture(GL_TEXTURE0 + textureUnit);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glUniform1i(glGetUniformLocation(program, uniformName), textureUnit);
}

static void _setTextureSamplers(GLuint program, const char uniformName[], size_t count, const int textureUnits[], const ovrAvatarAssetID assetIDs[])
{
	for (int i = 0; i < count; ++i)
	{
		ovrAvatarAssetID assetID = assetIDs[i];

		GLuint textureID = 0;
		if (assetID)
		{
			void* data = _assetMap[assetID];
			if (data)
			{
				TextureData* textureData = (TextureData*)data;
				textureID = textureData->textureID;
			}
		}
		glActiveTexture(GL_TEXTURE0 + textureUnits[i]);
		glBindTexture(GL_TEXTURE_2D, textureID);
	}
	GLint uniformLocation = glGetUniformLocation(program, uniformName);
	glUniform1iv(uniformLocation, (GLsizei)count, textureUnits);
}

static void _setMeshState(
	GLuint program,
	const ovrAvatarTransform& localTransform,
	const MeshData* data,
	const ovrAvatarSkinnedMeshPose& skinnedPose,
	const glm::mat4& world,
	const glm::mat4& view,
	const glm::mat4 proj,
	const glm::vec3& viewPos
) {
	// Compute the final world and viewProjection matrices for this part
	glm::mat4 local;
	_glmFromOvrAvatarTransform(localTransform, &local);
	glm::mat4 worldMat = world * local;
	glm::mat4 viewProjMat = proj * view;

	// Compute the skinned pose
	glm::mat4* skinnedPoses = (glm::mat4*)alloca(sizeof(glm::mat4) * skinnedPose.jointCount);
	_computeWorldPose(skinnedPose, skinnedPoses);
	for (uint32_t i = 0; i < skinnedPose.jointCount; ++i)
	{
		skinnedPoses[i] = skinnedPoses[i] * data->inverseBindPose[i];
	}

	// Pass the world view position to the shader for view-dependent rendering
	glUniform3fv(glGetUniformLocation(program, "viewPos"), 1, glm::value_ptr(viewPos));

	// Assign the vertex uniforms
	glUniformMatrix4fv(glGetUniformLocation(program, "world"), 1, 0, glm::value_ptr(worldMat));
	glUniformMatrix4fv(glGetUniformLocation(program, "viewProj"), 1, 0, glm::value_ptr(viewProjMat));
	glUniformMatrix4fv(glGetUniformLocation(program, "meshPose"), (GLsizei)skinnedPose.jointCount, 0, glm::value_ptr(*skinnedPoses));
}

static void _setMaterialState(GLuint program, const ovrAvatarMaterialState* state, glm::mat4* projectorInv)
{
	// Assign the fragment uniforms
	glUniform1i(glGetUniformLocation(program, "useAlpha"), state->alphaMaskTextureID != 0);
	glUniform1i(glGetUniformLocation(program, "useNormalMap"), state->normalMapTextureID != 0);
	glUniform1i(glGetUniformLocation(program, "useRoughnessMap"), state->roughnessMapTextureID != 0);

	glUniform1f(glGetUniformLocation(program, "elapsedSeconds"), _elapsedSeconds);

	if (projectorInv)
	{
		glUniform1i(glGetUniformLocation(program, "useProjector"), 1);
		glUniformMatrix4fv(glGetUniformLocation(program, "projectorInv"), 1, 0, glm::value_ptr(*projectorInv));
	}
	else
	{
		glUniform1i(glGetUniformLocation(program, "useProjector"), 0);
	}

	int textureSlot = 1;
	glUniform4fv(glGetUniformLocation(program, "baseColor"), 1, &state->baseColor.x);
	glUniform1i(glGetUniformLocation(program, "baseMaskType"), state->baseMaskType);
	glUniform4fv(glGetUniformLocation(program, "baseMaskParameters"), 1, &state->baseMaskParameters.x);
	glUniform4fv(glGetUniformLocation(program, "baseMaskAxis"), 1, &state->baseMaskAxis.x);
	_setTextureSampler(program, textureSlot++, "alphaMask", state->alphaMaskTextureID);
	glUniform4fv(glGetUniformLocation(program, "alphaMaskScaleOffset"), 1, &state->alphaMaskScaleOffset.x);
	_setTextureSampler(program, textureSlot++, "normalMap", state->normalMapTextureID);
	glUniform4fv(glGetUniformLocation(program, "normalMapScaleOffset"), 1, &state->normalMapScaleOffset.x);
	_setTextureSampler(program, textureSlot++, "parallaxMap", state->parallaxMapTextureID);
	glUniform4fv(glGetUniformLocation(program, "parallaxMapScaleOffset"), 1, &state->parallaxMapScaleOffset.x);
	_setTextureSampler(program, textureSlot++, "roughnessMap", state->roughnessMapTextureID);
	glUniform4fv(glGetUniformLocation(program, "roughnessMapScaleOffset"), 1, &state->roughnessMapScaleOffset.x);

	struct LayerUniforms {
		int layerSamplerModes[OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT];
		int layerBlendModes[OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT];
		int layerMaskTypes[OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT];
		ovrAvatarVector4f layerColors[OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT];
		int layerSurfaces[OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT];
		ovrAvatarAssetID layerSurfaceIDs[OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT];
		ovrAvatarVector4f layerSurfaceScaleOffsets[OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT];
		ovrAvatarVector4f layerSampleParameters[OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT];
		ovrAvatarVector4f layerMaskParameters[OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT];
		ovrAvatarVector4f layerMaskAxes[OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT];
	} layerUniforms;
	memset(&layerUniforms, 0, sizeof(layerUniforms));
	for (uint32_t i = 0; i < state->layerCount; ++i)
	{
		const ovrAvatarMaterialLayerState& layerState = state->layers[i];
		layerUniforms.layerSamplerModes[i] = layerState.sampleMode;
		layerUniforms.layerBlendModes[i] = layerState.blendMode;
		layerUniforms.layerMaskTypes[i] = layerState.maskType;
		layerUniforms.layerColors[i] = layerState.layerColor;
		layerUniforms.layerSurfaces[i] = textureSlot++;
		layerUniforms.layerSurfaceIDs[i] = layerState.sampleTexture;
		layerUniforms.layerSurfaceScaleOffsets[i] = layerState.sampleScaleOffset;
		layerUniforms.layerSampleParameters[i] = layerState.sampleParameters;
		layerUniforms.layerMaskParameters[i] = layerState.maskParameters;
		layerUniforms.layerMaskAxes[i] = layerState.maskAxis;
	}

	glUniform1i(glGetUniformLocation(program, "layerCount"), state->layerCount);
	glUniform1iv(glGetUniformLocation(program, "layerSamplerModes"), OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT, layerUniforms.layerSamplerModes);
	glUniform1iv(glGetUniformLocation(program, "layerBlendModes"), OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT, layerUniforms.layerBlendModes);
	glUniform1iv(glGetUniformLocation(program, "layerMaskTypes"), OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT, layerUniforms.layerMaskTypes);
	glUniform4fv(glGetUniformLocation(program, "layerColors"), OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT, (float*)layerUniforms.layerColors);
	_setTextureSamplers(program, "layerSurfaces", OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT, layerUniforms.layerSurfaces, layerUniforms.layerSurfaceIDs);
	glUniform4fv(glGetUniformLocation(program, "layerSurfaceScaleOffsets"), OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT, (float*)layerUniforms.layerSurfaceScaleOffsets);
	glUniform4fv(glGetUniformLocation(program, "layerSampleParameters"), OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT, (float*)layerUniforms.layerSampleParameters);
	glUniform4fv(glGetUniformLocation(program, "layerMaskParameters"), OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT, (float*)layerUniforms.layerMaskParameters);
	glUniform4fv(glGetUniformLocation(program, "layerMaskAxes"), OVR_AVATAR_MAX_MATERIAL_LAYER_COUNT, (float*)layerUniforms.layerMaskAxes);

}

static void _setPBSState(GLuint program, const ovrAvatarAssetID albedoTextureID, const ovrAvatarAssetID surfaceTextureID)
{
	int textureSlot = 0;
	_setTextureSampler(program, textureSlot++, "albedo", albedoTextureID);
	_setTextureSampler(program, textureSlot++, "surface", surfaceTextureID);
}

static void _renderDebugLine(const glm::mat4& worldViewProj, const glm::vec3& a, const glm::vec3& b, const glm::vec4& aColor, const glm::vec4& bColor)
{
	glUseProgram(_debugLineProgram);
	glUniformMatrix4fv(glGetUniformLocation(_debugLineProgram, "worldViewProj"), 1, 0, glm::value_ptr(worldViewProj));
	struct {
		glm::vec3 p;
		glm::vec4 c;
	} vertices[2] = {
		{ a, aColor },
		{ b, bColor },
	};

	glBindVertexArray(_debugVertexArray);
	glDepthFunc(GL_LEQUAL);
	glBindBuffer(GL_ARRAY_BUFFER, _debugVertexArray);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

	// Fill in the array attributes
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), 0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), (void*)sizeof(glm::vec3));
	glEnableVertexAttribArray(1);
	glLineWidth(25);
	glDrawArrays(GL_LINES, 0, 2);
}

static void _renderPose(const glm::mat4& worldViewProj, const ovrAvatarSkinnedMeshPose& pose, bool isRight)
{
	glm::mat4* skinnedPoses = (glm::mat4*)alloca(sizeof(glm::mat4) * pose.jointCount);
	_computeWorldPose(pose, skinnedPoses);
	if (!isRight) {
		_renderDebugLine(worldViewProj, glm::vec3(0, 0, 0), glm::vec3(0, 0, -200), laserColorLeft, glm::vec4(1, 1, 1, 1));
	}
	else {
		_renderDebugLine(worldViewProj, glm::vec3(0, 0, 0), glm::vec3(0, 0, -200), laserColorRight, glm::vec4(1, 1, 1, 1));
	}
	
	/*for (uint32_t i = 1; i < pose.jointCount; ++i)
	{
		int parent = pose.jointParents[i];
		_renderDebugLine(worldViewProj, glm::vec3(skinnedPoses[parent][3]), glm::vec3(skinnedPoses[i][3]), glm::vec4(1, 1, 1, 1), glm::vec4(1, 0, 0, 1));
	}*/
}

static void _renderSkinnedMeshPart(const ovrAvatarRenderPart_SkinnedMeshRender* mesh, uint32_t visibilityMask, const glm::mat4& world, const glm::mat4& view, const glm::mat4 proj, const glm::vec3& viewPos, bool isRight)
{
	// If this part isn't visible from the viewpoint we're rendering from, do nothing
	if ((mesh->visibilityMask & visibilityMask) == 0)
	{
		return;
	}

	// Get the GL mesh data for this mesh's asset
	MeshData* data = (MeshData*)_assetMap[mesh->meshAssetID];

	glUseProgram(_skinnedMeshProgram);

	// Apply the vertex state
	_setMeshState(_skinnedMeshProgram, mesh->localTransform, data, mesh->skinnedPose, world, view, proj, viewPos);

	// Apply the material state
	_setMaterialState(_skinnedMeshProgram, &mesh->materialState, nullptr);

	// Draw the mesh
	glBindVertexArray(data->vertexArray);
	glDepthFunc(GL_LEQUAL);

	// Write to depth first for self-occlusion
	if (mesh->visibilityMask & ovrAvatarVisibilityFlag_SelfOccluding)
	{
		//glDepthMask(GL_TRUE);
		//glColorMaski(0, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
		glDrawElements(GL_TRIANGLES, (GLsizei)data->elementCount, GL_UNSIGNED_SHORT, 0);
		//glDepthFunc(GL_EQUAL);
	}

	// Render to color buffer
	//glDepthMask(GL_FALSE);
	glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDrawElements(GL_TRIANGLES, (GLsizei)data->elementCount, GL_UNSIGNED_SHORT, 0);
	glBindVertexArray(0);

	/*if (renderJoints)
	{
		glm::mat4 local;
		_glmFromOvrAvatarTransform(mesh->localTransform, &local);
		glDepthFunc(GL_ALWAYS);
		_renderPose(proj * view * world * local, mesh->skinnedPose);
	}*/
	glm::mat4 local;
	_glmFromOvrAvatarTransform(mesh->localTransform, &local);
	glDepthFunc(GL_ALWAYS);
	_renderPose(proj * view * world * local, mesh->skinnedPose, isRight);
}

/* this part does not use */
static void _renderSkinnedMeshPartPBS(const ovrAvatarRenderPart_SkinnedMeshRenderPBS* mesh, uint32_t visibilityMask, const glm::mat4& world, const glm::mat4& view, const glm::mat4 proj, const glm::vec3& viewPos, bool isRight)
{
	// If this part isn't visible from the viewpoint we're rendering from, do nothing
	if ((mesh->visibilityMask & visibilityMask) == 0)
	{
		return;
	}

	// Get the GL mesh data for this mesh's asset
	MeshData* data = (MeshData*)_assetMap[mesh->meshAssetID];

	glUseProgram(_skinnedMeshPBSProgram);

	// Apply the vertex state
	_setMeshState(_skinnedMeshPBSProgram, mesh->localTransform, data, mesh->skinnedPose, world, view, proj, viewPos);

	// Apply the material state
	_setPBSState(_skinnedMeshPBSProgram, mesh->albedoTextureAssetID, mesh->surfaceTextureAssetID);

	// Draw the mesh
	glBindVertexArray(data->vertexArray);
	glDepthFunc(GL_LESS);

	// Write to depth first for self-occlusion
	if (mesh->visibilityMask & ovrAvatarVisibilityFlag_SelfOccluding)
	{
		glDepthMask(GL_TRUE);
		glColorMaski(0, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
		glDrawElements(GL_TRIANGLES, (GLsizei)data->elementCount, GL_UNSIGNED_SHORT, 0);
		glDepthFunc(GL_EQUAL);
	}
	glDepthMask(GL_FALSE);

	// Draw the mesh
	glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDrawElements(GL_TRIANGLES, (GLsizei)data->elementCount, GL_UNSIGNED_SHORT, 0);
	glBindVertexArray(0);

	/*if (renderJoints)
	{
		glm::mat4 local;
		_glmFromOvrAvatarTransform(mesh->localTransform, &local);
		glDepthFunc(GL_ALWAYS);
		_renderPose(proj * view * world * local, mesh->skinnedPose);
	}*/
	glm::mat4 local;
	_glmFromOvrAvatarTransform(mesh->localTransform, &local);
	glDepthFunc(GL_ALWAYS);
	_renderPose(proj * view * world * local, mesh->skinnedPose, isRight);
}

static void _renderProjector(const ovrAvatarRenderPart_ProjectorRender* projector, ovrAvatar* avatar, uint32_t visibilityMask, const glm::mat4& world, const glm::mat4& view, const glm::mat4 proj, const glm::vec3& viewPos)
{

	// Compute the mesh transform
	const ovrAvatarComponent* component = ovrAvatarComponent_Get(avatar, projector->componentIndex);
	const ovrAvatarRenderPart* renderPart = component->renderParts[projector->renderPartIndex];
	const ovrAvatarRenderPart_SkinnedMeshRender* mesh = ovrAvatarRenderPart_GetSkinnedMeshRender(renderPart);

	// If this part isn't visible from the viewpoint we're rendering from, do nothing
	if ((mesh->visibilityMask & visibilityMask) == 0)
	{
		return;
	}

	// Compute the projection matrix
	glm::mat4 projection;
	_glmFromOvrAvatarTransform(projector->localTransform, &projection);
	glm::mat4 worldProjection = world * projection;
	glm::mat4 projectionInv = glm::inverse(worldProjection);

	// Compute the mesh transform
	glm::mat4 meshWorld;
	_glmFromOvrAvatarTransform(component->transform, &meshWorld);

	// Get the GL mesh data for this mesh's asset
	MeshData* data = (MeshData*)_assetMap[mesh->meshAssetID];

	glUseProgram(_skinnedMeshProgram);

	// Apply the vertex state
	_setMeshState(_skinnedMeshProgram, mesh->localTransform, data, mesh->skinnedPose, meshWorld, view, proj, viewPos);

	// Apply the material state
	_setMaterialState(_skinnedMeshProgram, &projector->materialState, &projectionInv);

	// Draw the mesh
	glBindVertexArray(data->vertexArray);
	glDepthMask(GL_FALSE);
	glDepthFunc(GL_EQUAL);
	glDrawElements(GL_TRIANGLES, (GLsizei)data->elementCount, GL_UNSIGNED_SHORT, 0);
	glBindVertexArray(0);
}

static void _renderAvatar(ovrAvatar* avatar, uint32_t visibilityMask, const glm::mat4& view, const glm::mat4& proj, const glm::vec3& viewPos, bool renderJoints)
{
	// Traverse over all components on the avatar
	uint32_t componentCount = ovrAvatarComponent_Count(avatar);
	//printf("%zu", componentCount);
	for (uint32_t i = 4; i < 6; ++i)
	{
		const ovrAvatarComponent* component = ovrAvatarComponent_Get(avatar, i);
		bool isRight = false;

		//printf("%s\n", component->name);

		if (i == 5) {
			isRight = true;
		}

		// Compute the transform for this component
		glm::mat4 world;
		_glmFromOvrAvatarTransform(component->transform, &world);

		// Render each rebder part attached to the component
		for (uint32_t j = 0; j < component->renderPartCount; ++j)
		{
			
			const ovrAvatarRenderPart* renderPart = component->renderParts[j];
			ovrAvatarRenderPartType type = ovrAvatarRenderPart_GetType(renderPart);
			switch (type)
			{
			case ovrAvatarRenderPartType_SkinnedMeshRender:
				_renderSkinnedMeshPart(ovrAvatarRenderPart_GetSkinnedMeshRender(renderPart), visibilityMask, world, view, proj, viewPos, isRight);
				break;
			case ovrAvatarRenderPartType_SkinnedMeshRenderPBS:
				_renderSkinnedMeshPartPBS(ovrAvatarRenderPart_GetSkinnedMeshRenderPBS(renderPart), visibilityMask, world, view, proj, viewPos, isRight);
				break;
			case ovrAvatarRenderPartType_ProjectorRender:
				_renderProjector(ovrAvatarRenderPart_GetProjectorRender(renderPart), avatar, visibilityMask, world, view, proj, viewPos);
				break;
			}
		}
	}
}

static void _updateAvatar(
	ovrAvatar* avatar,
	float deltaSeconds,
	const ovrAvatarTransform& hmd,
	const ovrAvatarHandInputState& left,
	const ovrAvatarHandInputState& right,
	ovrMicrophone* mic,
	ovrAvatarPacket* packet,
	float* packetPlaybackTime
) {
	if (packet)
	{
		float packetDuration = ovrAvatarPacket_GetDurationSeconds(packet);
		*packetPlaybackTime += deltaSeconds;
		if (*packetPlaybackTime > packetDuration)
		{
			ovrAvatarPose_Finalize(avatar, 0.0f);
			*packetPlaybackTime = 0;
		}
		ovrAvatar_UpdatePoseFromPacket(avatar, packet, *packetPlaybackTime);
	}
	else
	{
		//// If we have a mic update the voice visualization
		//if (mic)
		//{
		//	float micSamples[48000];
		//	size_t sampleCount = ovr_Microphone_ReadData(mic, micSamples, sizeof(micSamples) / sizeof(micSamples[0]));
		//	if (sampleCount > 0)
		//	{
		//		ovrAvatarPose_UpdateVoiceVisualization(_avatar, (uint32_t)sampleCount, micSamples);
		//	}
		//}

		// Update the avatar pose from the inputs
		ovrAvatarPose_UpdateBody(avatar, hmd);
		ovrAvatarPose_UpdateHands(avatar, left, right);
	}
	ovrAvatarPose_Finalize(avatar, deltaSeconds);
}


/************************************************************************************
* OVR helpers
************************************************************************************/

//static ovrSession _initOVR()
//{
//	ovrSession ovr;
//	if (OVR_SUCCESS(ovr_Initialize(NULL)))
//	{
//		ovrGraphicsLuid luid;
//		if (OVR_SUCCESS(ovr_Create(&ovr, &luid)))
//		{
//			return ovr;
//		}
//		ovr_Shutdown();
//	}
//	return NULL;
//}
//
//static void _destroyOVR(ovrSession session)
//{
//	if (session)
//	{
//		ovr_Destroy(session);
//		ovr_Shutdown();
//	}
//}

/************************************************************************************
* Avatar message handlers
************************************************************************************/

static void _handleAvatarSpecification(const ovrAvatarMessage_AvatarSpecification* message)
{
	// Create the avatar instance
	_avatar = ovrAvatar_Create(message->avatarSpec, ovrAvatarCapability_All);

	// Trigger load operations for all of the assets referenced by the avatar
	uint32_t refCount = ovrAvatar_GetReferencedAssetCount(_avatar);
	for (uint32_t i = 0; i < refCount; ++i)
	{
		ovrAvatarAssetID id = ovrAvatar_GetReferencedAsset(_avatar, i);
		ovrAvatarAsset_BeginLoading(id);
		++_loadingAssets;
	}
	printf("Loading %d assets...\r\n", _loadingAssets);
}

static void _handleAssetLoaded(const ovrAvatarMessage_AssetLoaded* message)
{
	// Determine the type of the asset that got loaded
	ovrAvatarAssetType assetType = ovrAvatarAsset_GetType(message->asset);
	void* data = nullptr;

	// Call the appropriate loader function
	switch (assetType)
	{
	case ovrAvatarAssetType_Mesh:
		data = _loadMesh(ovrAvatarAsset_GetMeshData(message->asset));
		break;
	case ovrAvatarAssetType_Texture:
		data = _loadTexture(ovrAvatarAsset_GetTextureData(message->asset));
		break;
	}

	// Store the data that we loaded for the asset in the asset map
	_assetMap[message->assetID] = data;
	--_loadingAssets;
	printf("Loading %d assets...\r\n", _loadingAssets);
}

namespace ovr {

	// Convenience method for looping over each eye with a lambda
	template <typename Function>
	inline void for_each_eye(Function function) {
		for (ovrEyeType eye = ovrEyeType::ovrEye_Left;
			eye < ovrEyeType::ovrEye_Count;
			eye = static_cast<ovrEyeType>(eye + 1)) {
			function(eye);
		}
	}

	inline mat4 toGlm(const ovrMatrix4f & om) {
		return glm::transpose(glm::make_mat4(&om.M[0][0]));
	}

	inline mat4 toGlm(const ovrFovPort & fovport, float nearPlane = 0.01f, float farPlane = 10000.0f) {
		return toGlm(ovrMatrix4f_Projection(fovport, nearPlane, farPlane, true));
	}

	inline vec3 toGlm(const ovrVector3f & ov) {
		return glm::make_vec3(&ov.x);
	}

	inline vec2 toGlm(const ovrVector2f & ov) {
		return glm::make_vec2(&ov.x);
	}

	inline uvec2 toGlm(const ovrSizei & ov) {
		return uvec2(ov.w, ov.h);
	}

	inline quat toGlm(const ovrQuatf & oq) {
		return glm::make_quat(&oq.x);
	}

	inline mat4 toGlm(const ovrPosef & op) {
		mat4 orientation = glm::mat4_cast(toGlm(op.Orientation));
		mat4 translation = glm::translate(mat4(), ovr::toGlm(op.Position));
		return translation * orientation;
	}

	inline ovrMatrix4f fromGlm(const mat4 & m) {
		ovrMatrix4f result;
		mat4 transposed(glm::transpose(m));
		memcpy(result.M, &(transposed[0][0]), sizeof(float) * 16);
		return result;
	}

	inline ovrVector3f fromGlm(const vec3 & v) {
		ovrVector3f result;
		result.x = v.x;
		result.y = v.y;
		result.z = v.z;
		return result;
	}

	inline ovrVector2f fromGlm(const vec2 & v) {
		ovrVector2f result;
		result.x = v.x;
		result.y = v.y;
		return result;
	}

	inline ovrSizei fromGlm(const uvec2 & v) {
		ovrSizei result;
		result.w = v.x;
		result.h = v.y;
		return result;
	}

	inline ovrQuatf fromGlm(const quat & q) {
		ovrQuatf result;
		result.x = q.x;
		result.y = q.y;
		result.z = q.z;
		result.w = q.w;
		return result;
	}
}

class RiftManagerApp {
protected:
	ovrSession _session;
	ovrHmdDesc _hmdDesc;
	ovrGraphicsLuid _luid;

public:
	RiftManagerApp() {
		if (!OVR_SUCCESS(ovr_Create(&_session, &_luid))) {
			FAIL("Unable to create HMD session");
		}
		tempOvrSession = _session;

		_hmdDesc = ovr_GetHmdDesc(_session);
	}

	~RiftManagerApp() {
		ovr_Destroy(_session);
		_session = nullptr;
	}
};

class RiftApp : public GlfwApp, public RiftManagerApp {
public:

private:
	GLuint _fbo{ 0 };
	GLuint _depthBuffer{ 0 };
	ovrTextureSwapChain _eyeTexture;

	GLuint _mirrorFbo{ 0 };
	ovrMirrorTexture _mirrorTexture;

	ovrEyeRenderDesc _eyeRenderDescs[2];

	mat4 _eyeProjections[2];

	ovrLayerEyeFov _sceneLayer;
	ovrViewScaleDesc _viewScaleDesc;

	uvec2 _renderTargetSize;
	uvec2 _mirrorSize;

public:

	RiftApp() {
		using namespace ovr;
		_viewScaleDesc.HmdSpaceToWorldScaleInMeters = 1.0f;

		memset(&_sceneLayer, 0, sizeof(ovrLayerEyeFov));
		_sceneLayer.Header.Type = ovrLayerType_EyeFov;
		_sceneLayer.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;

		ovr::for_each_eye([&](ovrEyeType eye) {
			ovrEyeRenderDesc& erd = _eyeRenderDescs[eye] = ovr_GetRenderDesc(_session, eye, _hmdDesc.DefaultEyeFov[eye]);
			ovrMatrix4f ovrPerspectiveProjection =
				ovrMatrix4f_Projection(erd.Fov, 0.01f, 1000.0f, ovrProjection_ClipRangeOpenGL);
			_eyeProjections[eye] = ovr::toGlm(ovrPerspectiveProjection);
			_viewScaleDesc.HmdToEyeOffset[eye] = erd.HmdToEyeOffset;

			ovrFovPort & fov = _sceneLayer.Fov[eye] = _eyeRenderDescs[eye].Fov;
			auto eyeSize = ovr_GetFovTextureSize(_session, eye, fov, 1.0f);
			_sceneLayer.Viewport[eye].Size = eyeSize;
			_sceneLayer.Viewport[eye].Pos = { (int)_renderTargetSize.x, 0 };

			_renderTargetSize.y = std::max(_renderTargetSize.y, (uint32_t)eyeSize.h);
			_renderTargetSize.x += eyeSize.w;
		});
		// Make the on screen window 1/4 the resolution of the render target
		_mirrorSize = _renderTargetSize;
		_mirrorSize /= 4;
	}

protected:
	GLFWwindow * createRenderingTarget(uvec2 & outSize, ivec2 & outPosition) override {
		return glfw::createWindow(_mirrorSize);
	}

	void initGl() override {
		GlfwApp::initGl();

		// Compile the reference shaders
		char errorBuffer[512];
		_skinnedMeshProgram = _compileProgramFromFiles("AvatarVertexShader.glsl", "AvatarFragmentShader.glsl", sizeof(errorBuffer), errorBuffer);
		if (!_skinnedMeshProgram) {
			FAIL("Unable to _compileProgramFromFiles");
		}
		_skinnedMeshPBSProgram = _compileProgramFromFiles("AvatarVertexShader.glsl", "AvatarFragmentShaderPBS.glsl", sizeof(errorBuffer), errorBuffer);
		if (!_skinnedMeshPBSProgram) {
			FAIL("Unable to count swap chain textures");
		}

		const char debugLineVertexShader[] =
			"#version 330 core\n"
			"layout (location = 0) in vec3 position;\n"
			"layout (location = 1) in vec4 color;\n"
			"out vec4 vertexColor;\n"
			"uniform mat4 worldViewProj;\n"
			"void main() {\n"
			"    gl_Position = worldViewProj * vec4(position, 1.0);\n"
			"    vertexColor = color;\n"
			"}";

		const char debugLineFragmentShader[] =
			"#version 330 core\n"
			"in vec4 vertexColor;\n"
			"out vec4 fragmentColor;\n"
			"void main() {\n"
			"    fragmentColor = vertexColor;"
			"}";

		_debugLineProgram = _compileProgramFromSource(debugLineVertexShader, debugLineFragmentShader, sizeof(errorBuffer), errorBuffer);
		if (!_debugLineProgram) {
			FAIL("Unable to compile _debugLineProgram");
		}

		glGenVertexArrays(1, &_debugVertexArray);
		glGenBuffers(1, &_debugVertexBuffer);

		// Disable the v-sync for buffer swap
		glfwSwapInterval(0);

		ovrTextureSwapChainDesc desc = {};
		desc.Type = ovrTexture_2D;
		desc.ArraySize = 1;
		desc.Width = _renderTargetSize.x;
		desc.Height = _renderTargetSize.y;
		desc.MipLevels = 1;
		desc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
		desc.SampleCount = 1;
		desc.StaticImage = ovrFalse;
		ovrResult result = ovr_CreateTextureSwapChainGL(_session, &desc, &_eyeTexture);
		_sceneLayer.ColorTexture[0] = _eyeTexture;
		if (!OVR_SUCCESS(result)) {
			FAIL("Failed to create swap textures");
		}

		int length = 0;
		result = ovr_GetTextureSwapChainLength(_session, _eyeTexture, &length);
		if (!OVR_SUCCESS(result) || !length) {
			FAIL("Unable to count swap chain textures");
		}
		for (int i = 0; i < length; ++i) {
			GLuint chainTexId;
			ovr_GetTextureSwapChainBufferGL(_session, _eyeTexture, i, &chainTexId);
			glBindTexture(GL_TEXTURE_2D, chainTexId);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		}
		glBindTexture(GL_TEXTURE_2D, 0);

		// Set up the framebuffer object
		glGenFramebuffers(1, &_fbo);
		glGenRenderbuffers(1, &_depthBuffer);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _fbo);
		glBindRenderbuffer(GL_RENDERBUFFER, _depthBuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, _renderTargetSize.x, _renderTargetSize.y);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _depthBuffer);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

		ovrMirrorTextureDesc mirrorDesc;
		memset(&mirrorDesc, 0, sizeof(mirrorDesc));
		mirrorDesc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
		mirrorDesc.Width = _mirrorSize.x;
		mirrorDesc.Height = _mirrorSize.y;
		if (!OVR_SUCCESS(ovr_CreateMirrorTextureGL(_session, &mirrorDesc, &_mirrorTexture))) {
			FAIL("Could not create mirror texture");
		}
		glGenFramebuffers(1, &_mirrorFbo);
		lastTime = std::chrono::steady_clock::now();
	}

	void onKey(int key, int scancode, int action, int mods) override {
		if (GLFW_PRESS == action) switch (key) {
		case GLFW_KEY_R:
			ovr_RecenterTrackingOrigin(_session);
			return;
		}

		GlfwApp::onKey(key, scancode, action, mods);
	}

	void update() final override {
		while (ovrAvatarMessage* message = ovrAvatarMessage_Pop())
		{
			switch (ovrAvatarMessage_GetType(message))
			{
			case ovrAvatarMessageType_AvatarSpecification:
				_handleAvatarSpecification(ovrAvatarMessage_GetAvatarSpecification(message));
				break;
			case ovrAvatarMessageType_AssetLoaded:
				_handleAssetLoaded(ovrAvatarMessage_GetAssetLoaded(message));
				break;
			}
			ovrAvatarMessage_Free(message);
		}
	}

	void draw() final override {
		// Compute how much time has elapsed since the last frame
		std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
		std::chrono::duration<float> deltaTime = currentTime - lastTime;
		float deltaSeconds = deltaTime.count();
		lastTime = currentTime;
		_elapsedSeconds += deltaSeconds;

		ovrPosef eyePoses[2];
		ovr_GetEyePoses(_session, frame, true, _viewScaleDesc.HmdToEyeOffset, eyePoses, &_sceneLayer.SensorSampleTime);

		if (_avatar)
		{
			// Convert the OVR inputs into Avatar SDK inputs
			ovrInputState touchState;
			ovr_GetInputState(_session, ovrControllerType_Active, &touchState);
			ovrTrackingState trackingState = ovr_GetTrackingState(_session, 0.0, false);

			glm::vec3 hmdP = _glmFromOvrVector(trackingState.HeadPose.ThePose.Position);
			glm::quat hmdQ = _glmFromOvrQuat(trackingState.HeadPose.ThePose.Orientation);
			glm::vec3 leftP = _glmFromOvrVector(trackingState.HandPoses[ovrHand_Left].ThePose.Position);
			glm::quat leftQ = _glmFromOvrQuat(trackingState.HandPoses[ovrHand_Left].ThePose.Orientation);
			glm::vec3 rightP = _glmFromOvrVector(trackingState.HandPoses[ovrHand_Right].ThePose.Position);
			glm::quat rightQ = _glmFromOvrQuat(trackingState.HandPoses[ovrHand_Right].ThePose.Orientation);

			ovrAvatarTransform hmd;
			_ovrAvatarTransformFromGlm(hmdP, hmdQ, glm::vec3(1.0f), &hmd);

			ovrAvatarTransform left;
			_ovrAvatarTransformFromGlm(leftP, leftQ, glm::vec3(1.0f), &left);

			ovrAvatarTransform right;
			_ovrAvatarTransformFromGlm(rightP, rightQ, glm::vec3(1.0f), &right);

			ovrAvatarHandInputState inputStateLeft;
			_ovrAvatarHandInputStateFromOvr(left, touchState, ovrHand_Left, &inputStateLeft);

			ovrAvatarHandInputState inputStateRight;
			_ovrAvatarHandInputStateFromOvr(right, touchState, ovrHand_Right, &inputStateRight);

			_updateAvatar(_avatar, deltaSeconds, hmd, inputStateLeft, inputStateRight, nullptr, nullptr, 0);

			uint8_t amplitudeL = (uint8_t)round(inputStateLeft.indexTrigger * 150);
			uint8_t amplitudeR = (uint8_t)round(inputStateRight.indexTrigger * 150);

			left_line_pos = { vec3(inputStateLeft.transform.position.x,inputStateLeft.transform.position.y,inputStateLeft.transform.position.z), 
				trackingState.HandPoses[ovrHand_Left].ThePose.Orientation };
			right_line_pos = { vec3(inputStateRight.transform.position.x,inputStateRight.transform.position.y,inputStateRight.transform.position.z),
				trackingState.HandPoses[ovrHand_Right].ThePose.Orientation };

			if (inputStateLeft.buttonMask != 0 || inputStateRight.buttonMask != 0) {
				if (win || lost)
				{
					reset_flag = true;
				}
			}


			/*printf("\rleft_pos: %.6f, %.6f, %.6f     left_ori: %.6f, %.6f, %.6f, %.6f", inputStateLeft.transform.position.x, inputStateLeft.transform.position.y, inputStateLeft.transform.position.z,
				inputStateLeft.transform.orientation.x, inputStateLeft.transform.orientation.y, inputStateLeft.transform.orientation.z, inputStateLeft.transform.orientation.w);*/

			if (inputStateLeft.indexTrigger > 0.5) {
				//ovr_SetControllerVibration(_session, ovrControllerType_LTouch, inputStateLeft.indexTrigger * 0.3f, amplitudeL);
				laserColorLeft = glm::vec4(1, 0, 0, 1);
				left_trig = true;
			}
			else {
				ovr_SetControllerVibration(_session, ovrControllerType_LTouch, 0.0f, 0);
				laserColorLeft = glm::vec4(0, 1, 0, 1);
				left_trig = false;
			}
			if (inputStateRight.indexTrigger > 0.5) {
				//ovr_SetControllerVibration(_session, ovrControllerType_RTouch, inputStateRight.indexTrigger * 0.3f, amplitudeR);
				laserColorRight = glm::vec4(1, 0, 0, 1);
				right_trig = true;
			}
			else {
				ovr_SetControllerVibration(_session, ovrControllerType_RTouch, 0.0f, 0);
				laserColorRight = glm::vec4(0, 1, 0, 1);
				right_trig = false;
			}
		}

		int curIndex;
		ovr_GetTextureSwapChainCurrentIndex(_session, _eyeTexture, &curIndex);
		GLuint curTexId;
		ovr_GetTextureSwapChainBufferGL(_session, _eyeTexture, curIndex, &curTexId);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _fbo);
		glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, curTexId, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		ovr::for_each_eye([&](ovrEyeType eye) {
			const auto& vp = _sceneLayer.Viewport[eye];
			glViewport(vp.Pos.x, vp.Pos.y, vp.Size.w, vp.Size.h);
			_sceneLayer.RenderPose[eye] = eyePoses[eye];
			renderScene(_eyeProjections[eye], ovr::toGlm(eyePoses[eye]));

			ovrVector3f eyePosition = eyePoses[eye].Position;
			ovrQuatf eyeOrientation = eyePoses[eye].Orientation;
			glm::quat glmOrientation = _glmFromOvrQuat(eyeOrientation);
			glm::vec3 eyeWorld = _glmFromOvrVector(eyePosition);
			glm::vec3 eyeForward = glmOrientation * glm::vec3(0, 0, -1);
			glm::vec3 eyeUp = glmOrientation * glm::vec3(0, 1, 0);
			glm::mat4 view = glm::lookAt(eyeWorld, eyeWorld + eyeForward, eyeUp);

			ovrMatrix4f ovrProjection = ovrMatrix4f_Projection(_hmdDesc.DefaultEyeFov[eye], 0.01f, 1000.0f, ovrProjection_None);

			glm::mat4 proj(
				ovrProjection.M[0][0], ovrProjection.M[1][0], ovrProjection.M[2][0], ovrProjection.M[3][0],
				ovrProjection.M[0][1], ovrProjection.M[1][1], ovrProjection.M[2][1], ovrProjection.M[3][1],
				ovrProjection.M[0][2], ovrProjection.M[1][2], ovrProjection.M[2][2], ovrProjection.M[3][2],
				ovrProjection.M[0][3], ovrProjection.M[1][3], ovrProjection.M[2][3], ovrProjection.M[3][3]
			);

			// If we have the avatar and have finished loading assets, render it
			if (_avatar && !_loadingAssets)
			{
				_renderAvatar(_avatar, ovrAvatarVisibilityFlag_FirstPerson, view, proj, eyeWorld, false);

				glm::vec4 reflectionPlane = glm::vec4(0.0, 0.0, -1.0, 0.0);
				glm::mat4 reflection = _computeReflectionMatrix(reflectionPlane);

				glFrontFace(GL_CW);
				//_renderAvatar(_avatar, ovrAvatarVisibilityFlag_ThirdPerson, view * reflection, proj, glm::vec3(reflection * glm::vec4(eyeWorld, 1.0f)), false);
				glFrontFace(GL_CCW);
			}
		});
		glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		ovr_CommitTextureSwapChain(_session, _eyeTexture);
		ovrLayerHeader* headerList = &_sceneLayer.Header;
		ovr_SubmitFrame(_session, frame, &_viewScaleDesc, &headerList, 1);

		GLuint mirrorTextureId;
		ovr_GetMirrorTextureBufferGL(_session, _mirrorTexture, &mirrorTextureId);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, _mirrorFbo);
		glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mirrorTextureId, 0);
		glBlitFramebuffer(0, 0, _mirrorSize.x, _mirrorSize.y, 0, _mirrorSize.y, _mirrorSize.x, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	}

	virtual void renderScene(const glm::mat4 & projection, const glm::mat4 & headPose) = 0;
};

//////////////////////////////////////////////////////////////////////
//
// The remainder of this code is specific to the scene we want to 
// render.  I use oglplus to render an array of cubes, but your 
// application would perform whatever rendering you want
//


//////////////////////////////////////////////////////////////////////
//
// OGLplus is a set of wrapper classes for giving OpenGL a more object
// oriented interface
//
#define OGLPLUS_USE_GLCOREARB_H 0
#define OGLPLUS_USE_GLEW 1
#define OGLPLUS_USE_BOOST_CONFIG 0
#define OGLPLUS_NO_SITE_CONFIG 1
#define OGLPLUS_LOW_PROFILE 1

#pragma warning( disable : 4068 4244 4267 4065)
#include <oglplus/config/basic.hpp>
#include <oglplus/config/gl.hpp>
#include <oglplus/all.hpp>
#include <oglplus/interop/glm.hpp>
#include <oglplus/bound/texture.hpp>
#include <oglplus/bound/framebuffer.hpp>
#include <oglplus/bound/renderbuffer.hpp>
#include <oglplus/bound/buffer.hpp>
#include <oglplus/shapes/cube.hpp>
#include <oglplus/shapes/wrapper.hpp>
#pragma warning( default : 4068 4244 4267 4065)



namespace Attribute {
	enum {
		Position = 0,
		TexCoord0 = 1,
		Normal = 2,
		Color = 3,
		TexCoord1 = 4,
		InstanceTransform = 5,
	};
}

static const char * VERTEX_SHADER = R"SHADER(
#version 410 core

uniform mat4 ProjectionMatrix = mat4(1);
uniform mat4 CameraMatrix = mat4(1);

layout(location = 0) in vec4 Position;
layout(location = 2) in vec3 Normal;
layout(location = 5) in mat4 InstanceTransform;

out vec3 vertNormal;

void main(void) {
   mat4 ViewXfm = CameraMatrix * InstanceTransform;
   //mat4 ViewXfm = CameraMatrix;
   vertNormal = Normal;
   gl_Position = ProjectionMatrix * ViewXfm * Position;
}
)SHADER";

static const char * FRAGMENT_SHADER = R"SHADER(
#version 410 core

in vec3 vertNormal;
out vec4 fragColor;

void main(void) {
    vec3 color = vertNormal;
    if (!all(equal(color, abs(color)))) {
        color = vec3(1.0) - abs(color);
    }
    fragColor = vec4(color, 1.0);
}
)SHADER";

// a class for encapsulating building and rendering an RGB cube
struct ColorCubeScene {

	// Program
	//oglplus::shapes::ShapeWrapper cube;
	

	Model fac1;
	Model co2_tmp;
	Model o2_tmp;
	vector<Model> co2_arr;
	std::vector<mat4> co2_pos;
	std::vector<Model> o2_arr;
	std::clock_t start;
	double duration;
	vector<vector<double>> velocity;
	vector<double> rotata;
	vector<double> rotatas;
	vector<vec3> rotata_ang;
	int count_o2 = 0;

	vector<mat4> los_pos;

	// VBOs for the cube's vertices and normals

	const unsigned int GRID_SIZE{ 5 };

public:
	ColorCubeScene(){

		fac1 = Model("C:\\Users\\zyc19\\Downloads\\RobinCS190-all\\RobinCS190\\RobinCS190\\MinimalVR-master\\Minimal\\factory1.obj", "CO2");
		co2_tmp = Model("C:\\Users\\zyc19\\Downloads\\RobinCS190-all\\RobinCS190\\RobinCS190\\MinimalVR-master\\Minimal\\co2.obj", "CO2");
		o2_tmp = Model("C:\\Users\\zyc19\\Downloads\\RobinCS190-all\\RobinCS190\\RobinCS190\\MinimalVR-master\\Minimal\\o2.obj", "O2");

		for (int i = 0; i < 5; i++)
		{
			co2_arr.push_back(co2_tmp);
			float xpos = -0.7f + (rand()) / (float)(RAND_MAX / 1.4f);
			float ypos = -1.0f + (rand()) / (float)(RAND_MAX);
			float zpos = -2.4f + (rand()) / (float)(RAND_MAX / 2.0f);
			vec3 relativePosition = vec3(xpos, ypos, zpos);

			co2_pos.push_back(glm::translate(glm::mat4(1.0f), relativePosition));
			/*if (relativePosition == vec3(0)) {
				continue;
			}*/
			
			double v1 = 0.0003f + (rand()) / (float)(RAND_MAX / (0.0008f - 0.0003f));
			double v2 = 0.0003f + (rand()) / (float)(RAND_MAX / (0.0008f - 0.0003f));
			double v3 = 0.0003f + (rand()) / (float)(RAND_MAX / (0.0008f - 0.0003f));

			if (rand() / (float)(RAND_MAX) > 0.5f)
			{
				v1 *= -1.0f;
			}
			if (rand() / (float)(RAND_MAX) > 0.5f)
			{
				v2 *= -1.0f;
			}
			if (rand() / (float)(RAND_MAX) > 0.5f)
			{
				v3 *= -1.0f;
			}
			vector<double> tmp = { v1,v2,v3 };
			velocity.push_back(tmp);


			double r = 0.01f + (rand()) / (float)(RAND_MAX / (0.02f - 0.01f));
			rotata.push_back(r);
			rotatas.push_back(r);
			rotata_ang.push_back(vec3((rand()) / (float)(RAND_MAX), (rand()) / (float)(RAND_MAX), (rand()) / (float)(RAND_MAX)));
		}

		
		for (int i = 0; i < 100; i++)
		{
			float xpos = -0.7f + (rand()) / (float)(RAND_MAX / 1.4f);
			float ypos = -1.0f + (rand()) / (float)(RAND_MAX);
			float zpos = -2.4f + (rand()) / (float)(RAND_MAX / 2.0f);
			vec3 relativePosition = vec3(xpos, ypos, zpos);
			float r = (rand()) / (float)(RAND_MAX / 10.0f);
			
			
			los_pos.push_back(glm::rotate(glm::translate(glm::mat4(1.0f), relativePosition), r, vec3((rand()) / (float)(RAND_MAX), (rand()) / (float)(RAND_MAX), (rand()) / (float)(RAND_MAX))));
		}
		start = clock();


	}

	void render(const mat4 & projection, const mat4 & modelview) {
		Shader sd("./shader.vert", "./shader.frag");

		if (o2_arr.size() == co2_arr.size())
		{
			win = true;
			glClearColor(0.0f, 0.73f, 1.0f, 0.0f);
		}

		if (co2_arr.size() - o2_arr.size() >= 10)
		{
			lost = true;
			//glClearColor(0.3f, 0.0f, 0.0f, 0.0f);
		}

		sd.Use();

		glUniformMatrix4fv(glGetUniformLocation(sd.Program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
		glUniformMatrix4fv(glGetUniformLocation(sd.Program, "view"), 1, GL_FALSE, glm::value_ptr(modelview));
		glm::mat4 mod;
		mod = glm::translate(mod, glm::vec3(0.0f, -0.8f, -2.0f));
		mod = glm::scale(mod, glm::vec3(0.05f, 0.05f, 0.05f));
		glUniformMatrix4fv(glGetUniformLocation(sd.Program, "model"), 1, GL_FALSE, glm::value_ptr(mod));
		glUniformMatrix4fv(glGetUniformLocation(sd.Program, "viewPos"), 1, GL_FALSE, glm::value_ptr(modelview));

		GLint lightAmbientLoc = glGetUniformLocation(sd.Program, "light.ambient");
		GLint lightDiffuseLoc = glGetUniformLocation(sd.Program, "light.diffuse");
		GLint lightSpecularLoc = glGetUniformLocation(sd.Program, "light.specular");
		GLint lightPos = glGetUniformLocation(sd.Program, "light.position");
		glUniform3f(lightAmbientLoc, 0.2f, 0.2f, 0.2f);
		glUniform3f(lightDiffuseLoc, 1.0f, 1.0f, 1.0f); // Let's darken the light a bit to fit the scene
		glUniform3f(lightSpecularLoc, 1.0f, 1.0f, 1.0f);
		glUniform3f(lightPos, 1.0f, 1.0f, 1.0f);

		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

		if (duration > 1.5f && !win && !lost)
		{
			co2_arr.push_back(co2_tmp);
			duration = 0;
			double v1 = 0.0003f + (rand()) / (float)(RAND_MAX / (0.0008f - 0.0003f));
			double v2 = 0.0003f + (rand()) / (float)(RAND_MAX / (0.0008f - 0.0003f));
			double v3 = 0.0003f + (rand()) / (float)(RAND_MAX / (0.0008f - 0.0003f));
			if (rand() / (float)(RAND_MAX) > 0.5f)
			{
				v1 *= -1.0f;
			}
			if (rand() / (float)(RAND_MAX) > 0.5f)
			{
				v2 *= -1.0f;
			}
			if (rand() / (float)(RAND_MAX) > 0.5f)
			{
				v3 *= -1.0f;
			}
			vector<double> tmp = { v1,v2,v3 };
			velocity.push_back(tmp);

			double r = 0.01f + (rand()) / (float)(RAND_MAX / (0.02f - 0.01f));
			rotata.push_back(r);
			rotatas.push_back(r);
			rotata_ang.push_back(vec3((rand()) / (float)(RAND_MAX), (rand()) / (float)(RAND_MAX), (rand()) / (float)(RAND_MAX)));
			co2_pos.push_back(glm::scale(mod, glm::vec3(20.0f, 20.0f, 20.0f)));
			start = clock();
		}

		fac1.Draw(sd);

		if (lost)
		{
			glUniformMatrix4fv(glGetUniformLocation(sd.Program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
			glUniformMatrix4fv(glGetUniformLocation(sd.Program, "view"), 1, GL_FALSE, glm::value_ptr(modelview));
			glUniformMatrix4fv(glGetUniformLocation(sd.Program, "viewPos"), 1, GL_FALSE, glm::value_ptr(modelview));
			glm::mat4 mod = mat4();
			for (int i = 0; i < los_pos.size(); i++)
			{
				mod = los_pos[i];
				mod = glm::scale(mod, glm::vec3(0.05f, 0.05f, 0.05f));
				glUniformMatrix4fv(glGetUniformLocation(sd.Program, "model"), 1, GL_FALSE, glm::value_ptr(mod));
				co2_tmp.Draw(sd);
			}
		}
		else
		{
			for (int i = 0; i < co2_arr.size(); i++)
			{
				glUniformMatrix4fv(glGetUniformLocation(sd.Program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
				glUniformMatrix4fv(glGetUniformLocation(sd.Program, "view"), 1, GL_FALSE, glm::value_ptr(modelview));
				glUniformMatrix4fv(glGetUniformLocation(sd.Program, "viewPos"), 1, GL_FALSE, glm::value_ptr(modelview));

				if (win)
				{
					char buff[100];
					sprintf_s(buff, "the %d object\n", i);
					OutputDebugStringA(buff);
					glm::mat4 mod = co2_pos[i];
					glUniformMatrix4fv(glGetUniformLocation(sd.Program, "model"), 1, GL_FALSE, glm::value_ptr(mod));
					co2_arr[i].Draw(sd);
				}
				else
				{
					glm::mat4 mod = mat4();

					mod = glm::translate(mod, glm::vec3(velocity[i][0], velocity[i][1], velocity[i][2]));
					mod = glm::translate(mod, vec3(co2_pos[i][3][0], co2_pos[i][3][1], co2_pos[i][3][2]));
					mod = glm::rotate(mod, (float)rotata[i], rotata_ang[i]);
					mod = glm::scale(mod, glm::vec3(0.05f, 0.05f, 0.05f));

					co2_pos[i] = mod;

					if (mod[3][0] >= 1.0f || mod[3][0] <= -1.0f)
					{
						velocity[i][0] *= -1.0f;
					}

					if (mod[3][1] >= 0.3f || mod[3][1] <= -1.2f)
					{
						velocity[i][1] *= -1.0f;
					}

					if (mod[3][2] >= -0.8f || mod[3][2] <= -3.0f)
					{
						velocity[i][2] *= -1.0f;
					}

					rotata[i] += rotatas[i];

					glm::quat qut_L = quat(left_line_pos.second.w, left_line_pos.second.x, left_line_pos.second.y, left_line_pos.second.z);
					vec3 dir_vect = qut_L * vec3(0.0f, 0.0f, -1.0f);
					vec3 endPoint = dir_vect * 100000.0f;
					vec3 nextDist = (endPoint - left_line_pos.first);
					vec3 tmp = glm::cross(nextDist, (left_line_pos.first - vec3(mod[3])));
					double tmp2 = sqrt(pow(nextDist.x, 2) + pow(nextDist.y, 2) + pow(nextDist.z, 2));
					double dist_L = sqrt(pow(tmp.x, 2) + pow(tmp.y, 2) + pow(tmp.z, 2)) / tmp2;

					glm::quat qut_R = quat(right_line_pos.second.w, right_line_pos.second.x, right_line_pos.second.y, right_line_pos.second.z);
					dir_vect = qut_R * vec3(0.0f, 0.0f, -1.0f);
					endPoint = dir_vect * 100000.0f;
					nextDist = (endPoint - right_line_pos.first);
					tmp = glm::cross(nextDist, (right_line_pos.first - vec3(mod[3])));
					tmp2 = sqrt(pow(nextDist.x, 2) + pow(nextDist.y, 2) + pow(nextDist.z, 2));
					double dist_R = sqrt(pow(tmp.x, 2) + pow(tmp.y, 2) + pow(tmp.z, 2)) / tmp2;

					if (!co2_arr[i].is_O2() && dist_R <= 0.06f && dist_L <= 0.06f && left_trig && right_trig)
					{
						ovr_SetControllerVibration(tempOvrSession, ovrControllerType_LTouch, 1.0f, 255);
						ovr_SetControllerVibration(tempOvrSession, ovrControllerType_RTouch, 1.0f, 255);
						co2_arr[i] = o2_tmp;
						o2_arr.push_back(o2_tmp);
						char buff[100];
						sprintf_s(buff, "the %d object intersect\n", i);
						OutputDebugStringA(buff);
					}

					glUniformMatrix4fv(glGetUniformLocation(sd.Program, "model"), 1, GL_FALSE, glm::value_ptr(mod));
					co2_arr[i].Draw(sd);
				}
			}
		}

		

	}
};


// An example application that renders a simple cube
class ExampleApp : public RiftApp {
	std::shared_ptr<ColorCubeScene> cubeScene;

public:
	/*
	Model fac1;
	Model co2_tmp;
	Model o2_tmp;
	vector<Model> co2_arr;
	std::vector<mat4> co2_pos;
	std::vector<Model> o2_arr;
	std::clock_t start;
	double duration;
	vector<vector<double>> velocity;
	vector<double> rotata;
	vector<double> rotatas;
	vector<vec3> rotata_ang;
	int count_o2 = 0;
	*/
	ExampleApp() {}
protected:
	void initGl() override {
		RiftApp::initGl();
		glClearColor(0.0f, 0.0f, 0.55f, 0.0f);
		glEnable(GL_CULL_FACE);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glClearDepth(1.0f);

		// Initialize the avatar module
		ovrAvatar_Initialize(MIRROR_SAMPLE_APP_ID);

		// Start retrieving the avatar specification
		printf("Requesting avatar specification...\r\n");
		ovrID userID = ovr_GetLoggedInUserID();
		ovrAvatar_RequestAvatarSpecification(userID);

		// Recenter the tracking origin at startup so that the reflection avatar appears directly in front of the user
		ovr_RecenterTrackingOrigin(_session);


		/*
		fac1 = Model("C:\\Users\\yuz287\\Downloads\\MinimalVR-master\\Minimal\\factory1.obj", "CO2");
		co2_tmp = Model("C:\\Users\\yuz287\\Downloads\\MinimalVR-master\\Minimal\\co2.obj", "CO2");
		o2_tmp = Model("C:\\Users\\yuz287\\Downloads\\MinimalVR-master\\Minimal\\o2.obj", "O2");
		// Initialize the avatar module
		ovrAvatar_Initialize(MIRROR_SAMPLE_APP_ID);

		// Start retrieving the avatar specification
		printf("Requesting avatar specification...\r\n");
		ovrID userID = ovr_GetLoggedInUserID();
		ovrAvatar_RequestAvatarSpecification(userID);

		// Recenter the tracking origin at startup so that the reflection avatar appears directly in front of the user
		ovr_RecenterTrackingOrigin(_session);

		for (int i = 0; i < 5; i++)
		{
			co2_arr.push_back(co2_tmp);
			float xpos = (rand()) / (float)(RAND_MAX) / 2.0f;
			float ypos = (rand()) / (float)(RAND_MAX) / 1.4f;
			float zpos = (rand()) / (float)(RAND_MAX) / 1.4f;
			vec3 relativePosition = vec3(xpos, ypos, -zpos);
			if (relativePosition == vec3(0)) {
				continue;
			}
			co2_pos.push_back(glm::translate(glm::mat4(1.0f), relativePosition));


			double v1 = 0.0003f + (rand()) / (float)(RAND_MAX / (0.0008f - 0.0003f));
			double v2 = 0.0003f + (rand()) / (float)(RAND_MAX / (0.0008f - 0.0003f));
			double v3 = 0.0003f + (rand()) / (float)(RAND_MAX / (0.0008f - 0.0003f));

			if (rand() / (float)(RAND_MAX) > 0.5f)
			{
				v1 *= -1.0f;
			}
			if (rand() / (float)(RAND_MAX) > 0.5f)
			{
				v2 *= -1.0f;
			}
			if (rand() / (float)(RAND_MAX) > 0.5f)
			{
				v3 *= -1.0f;
			}
			vector<double> tmp = { v1,v2,v3 };
			velocity.push_back(tmp);


			double r = 0.01f + (rand()) / (float)(RAND_MAX / (0.02f - 0.01f));
			rotata.push_back(r);
			rotatas.push_back(r);
			rotata_ang.push_back(vec3((rand()) / (float)(RAND_MAX), (rand()) / (float)(RAND_MAX), (rand()) / (float)(RAND_MAX)));
		}
		start = clock();
		*/

		cubeScene = std::shared_ptr<ColorCubeScene>(new ColorCubeScene());
	}

	void shutdownGl() override {
		cubeScene.reset();
	}

	void renderScene(const glm::mat4 & projection, const glm::mat4 & headPose) override {
		if (win || lost)
		{
			if (reset_flag)
			{
				win = false;
				lost = false;
				reset_flag = false;
				glClearColor(0.0f, 0.0f, 0.55f, 0.0f);
				cubeScene = std::shared_ptr<ColorCubeScene>(new ColorCubeScene());
			}
		}
		cubeScene->render(projection, glm::inverse(headPose));

		/*
		Shader sd("./shader.vert", "./shader.frag");

		if (o2_arr.size() == co2_arr.size())
		{
			win = true;
			glClearColor(0.0f, 0.0f, 0.85f, 0.0f);
		}

		if (co2_arr.size() - o2_arr.size() >= 10)
		{
			lost = true;
			glClearColor(0.3f, 0.0f, 0.0f, 0.0f);
		}

		sd.Use();

		glUniformMatrix4fv(glGetUniformLocation(sd.Program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
		glUniformMatrix4fv(glGetUniformLocation(sd.Program, "view"), 1, GL_FALSE, glm::value_ptr(glm::inverse(headPose)));
		glm::mat4 mod;
		mod = glm::translate(mod, glm::vec3(0.0f, -0.4f, -1.0f));
		mod = glm::scale(mod, glm::vec3(0.05f, 0.05f, 0.05f));
		glUniformMatrix4fv(glGetUniformLocation(sd.Program, "model"), 1, GL_FALSE, glm::value_ptr(mod));
		glUniformMatrix4fv(glGetUniformLocation(sd.Program, "viewPos"), 1, GL_FALSE, glm::value_ptr(glm::inverse(headPose)));

		GLint lightAmbientLoc = glGetUniformLocation(sd.Program, "light.ambient");
		GLint lightDiffuseLoc = glGetUniformLocation(sd.Program, "light.diffuse");
		GLint lightSpecularLoc = glGetUniformLocation(sd.Program, "light.specular");
		GLint lightPos = glGetUniformLocation(sd.Program, "light.position");
		glUniform3f(lightAmbientLoc, 0.2f, 0.2f, 0.2f);
		glUniform3f(lightDiffuseLoc, 1.0f, 1.0f, 1.0f); // Let's darken the light a bit to fit the scene
		glUniform3f(lightSpecularLoc, 1.0f, 1.0f, 1.0f);
		glUniform3f(lightPos, 1.0f, 1.0f, 1.0f);

		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

		if (duration > 3 && !win && !lost)
		{
			co2_arr.push_back(co2_tmp);
			duration = 0;
			double v1 = 0.0003f + (rand()) / (float)(RAND_MAX / (0.0008f - 0.0003f));
			double v2 = 0.0003f + (rand()) / (float)(RAND_MAX / (0.0008f - 0.0003f));
			double v3 = 0.0003f + (rand()) / (float)(RAND_MAX / (0.0008f - 0.0003f));
			if (rand() / (float)(RAND_MAX) > 0.5f)
			{
				v1 *= -1.0f;
			}
			if (rand() / (float)(RAND_MAX) > 0.5f)
			{
				v2 *= -1.0f;
			}
			if (rand() / (float)(RAND_MAX) > 0.5f)
			{
				v3 *= -1.0f;
			}
			vector<double> tmp = { v1,v2,v3 };
			velocity.push_back(tmp);

			double r = 0.01f + (rand()) / (float)(RAND_MAX / (0.02f - 0.01f));
			rotata.push_back(r);
			rotatas.push_back(r);
			rotata_ang.push_back(vec3((rand()) / (float)(RAND_MAX), (rand()) / (float)(RAND_MAX), (rand()) / (float)(RAND_MAX)));
			co2_pos.push_back(glm::scale(mod, glm::vec3(20.0f, 20.0f, 20.0f)));
			start = clock();
		}

		fac1.Draw(sd);

		for (int i = 0; i < co2_arr.size(); i++)
		{
			glUniformMatrix4fv(glGetUniformLocation(sd.Program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
			glUniformMatrix4fv(glGetUniformLocation(sd.Program, "view"), 1, GL_FALSE, glm::value_ptr(glm::inverse(headPose)));
			glUniformMatrix4fv(glGetUniformLocation(sd.Program, "viewPos"), 1, GL_FALSE, glm::value_ptr(glm::inverse(headPose)));

			if (win || lost)
			{
				char buff[100];
				sprintf_s(buff, "the %d object\n", i);
				OutputDebugStringA(buff);
				glm::mat4 mod = co2_pos[i];
				glUniformMatrix4fv(glGetUniformLocation(sd.Program, "model"), 1, GL_FALSE, glm::value_ptr(mod));
				co2_arr[i].Draw(sd);
			}
			else
			{
				glm::mat4 mod = mat4();

				mod = glm::translate(mod, glm::vec3(velocity[i][0], velocity[i][1], velocity[i][2]));
				mod = glm::translate(mod, vec3(co2_pos[i][3][0], co2_pos[i][3][1], co2_pos[i][3][2]));
				mod = glm::rotate(mod, (float)rotata[i], rotata_ang[i]);
				mod = glm::scale(mod, glm::vec3(0.05f, 0.05f, 0.05f));

				co2_pos[i] = mod;

				if (mod[3][0] >= 0.8f || mod[3][0] <= -0.5f)
				{
					velocity[i][0] *= -1.0f;
				}

				if (mod[3][1] >= 0.5f || mod[3][1] <= -0.6f)
				{
					velocity[i][1] *= -1.0f;
				}

				if (mod[3][2] >= 0.5f || mod[3][2] <= -1.3f)
				{
					velocity[i][2] *= -1.0f;
				}

				rotata[i] += rotatas[i];

				glm::quat qut_L = quat(left_line_pos.second.w, left_line_pos.second.x, left_line_pos.second.y, left_line_pos.second.z);
				vec3 dir_vect = qut_L * vec3(0.0f, 0.0f, -1.0f);
				vec3 endPoint = dir_vect * 10000.0f;
				vec3 nextDist = (endPoint - left_line_pos.first);
				vec3 tmp = glm::cross(nextDist, (left_line_pos.first - vec3(mod[3])));
				double tmp2 = sqrt(pow(nextDist.x, 2) + pow(nextDist.y, 2) + pow(nextDist.z, 2));
				double dist_L = sqrt(pow(tmp.x, 2) + pow(tmp.y, 2) + pow(tmp.z, 2)) / tmp2;

				glm::quat qut_R = quat(right_line_pos.second.w, right_line_pos.second.x, right_line_pos.second.y, right_line_pos.second.z);
				dir_vect = qut_R * vec3(0.0f, 0.0f, -1.0f);
				endPoint = dir_vect * 10000.0f;
				nextDist = (endPoint - left_line_pos.first);
				tmp = glm::cross(nextDist, (left_line_pos.first - vec3(mod[3])));
				tmp2 = sqrt(pow(nextDist.x, 2) + pow(nextDist.y, 2) + pow(nextDist.z, 2));
				double dist_R = sqrt(pow(tmp.x, 2) + pow(tmp.y, 2) + pow(tmp.z, 2)) / tmp2;

				if (!co2_arr[i].is_O2() &&dist_R <= 0.08f && dist_L <= 0.08f && left_trig && right_trig)
				{
					co2_arr[i] = o2_tmp;
					o2_arr.push_back(o2_tmp);
					char buff[100];
					sprintf_s(buff, "the %d object intersect\n", i);
					OutputDebugStringA(buff);
				}

				glUniformMatrix4fv(glGetUniformLocation(sd.Program, "model"), 1, GL_FALSE, glm::value_ptr(mod));
				co2_arr[i].Draw(sd);
			}
		}

		*/
	}
};

// Execute our example class
int __stdcall WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	int result = -1;
	AllocConsole();
	freopen("conin$", "r", stdin);
	freopen("conout$", "w", stdout);
	freopen("conout$", "w", stderr);
	try {
		// Initialization call
		if (ovr_PlatformInitializeWindows(MIRROR_SAMPLE_APP_ID) != ovrPlatformInitialize_Success)
		{
			FAIL("Failed to initialize the Oculus Platform");
			// Exit.  Initialization failed which means either the oculus service isn¡¯t on the machine or they¡¯ve hacked their DLL
		}
		ovr_Entitlement_GetIsViewerEntitled();
		if (!OVR_SUCCESS(ovr_Initialize(nullptr))) {
			FAIL("Failed to initialize the Oculus SDK");
		}

		result = ExampleApp().run();

	}
	catch (std::exception & error) {
		OutputDebugStringA(error.what());
		std::cerr << error.what() << std::endl;
	}
	ovr_Shutdown();
	return result;
}