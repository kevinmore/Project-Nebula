#pragma once
#include <Utility/DataTypes.h>


class Camera
{
private:
	vec3 m_position;
	vec3 m_center;
	vec3 m_viewDirection;
	vec3 m_strafeDirection;
	const vec3 m_Up;
	
	vec2 m_lastMousePosition;
	const float rotationSpeed;

public:
	float movementSpeed;
	Camera(void) : m_position(0.0f, 0.1f, 0.2f), 
				   m_center(0.0f, 0.0f, 0.0f), 
				   m_viewDirection(0.0f, -0.2f, -1.0f),
				   m_Up(0.0f, 1.0f, 0.0f),
				   movementSpeed(0.01f),
				   rotationSpeed(0.2f)
	
	{};
	~Camera(void);

	vec3 Position() const { return m_position; }
	void Position(vec3 val) { m_position = val; }
	vec3 Center() const { return m_center; }
	void Center(vec3 val) { m_center = val; }
	const vec3 Up() const { return m_Up; }
	vec3 ViewDirection() const { return m_viewDirection; }
	void ViewDirection(vec3 val) { m_viewDirection = val; }
	vec2 LastMousePosition() const { return m_lastMousePosition; }
	void LastMousePosition(vec2 val) { m_lastMousePosition = val; }
	void lookAround(const vec2& currentMousePosition);
	void moveForward();
	void moveBackward();
	void moveLeft();
	void moveRight();
	void moveUp();
	void moveDown();
};

