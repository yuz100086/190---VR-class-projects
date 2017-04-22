#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoords;

out vec3 FragPos;
out vec3 vertNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0f);
	vertNormal = normal;
    FragPos = vec3(texCoords, 1.0f);
}

/*
#version 410 core

uniform mat4 projection = mat4(1);
uniform mat4 view = mat4(1);
uniform mat4 model = mat4(1);
//uniform vec4 Position = vec4(1);

layout(location = 0) in vec4 Position;
layout(location = 2) in vec3 Normal;
//layout(location = 5) in mat4 model;

out vec3 FragPos;
out vec3 vertNormal;

void main(void) {
   mat4 ViewXfm = view * model;
   //mat4 ViewXfm = view;
   vertNormal = Normal;
   gl_Position = projection * ViewXfm * Position;
   FragPos = vec3(gl_Position);
}
*/