#include <Utility/Camera.h>

Camera::~Camera(void)
{
}



void Camera::lookAround( const vec2& currentMousePosition )
{
	// calculate the mouse movement
	vec2 mouseDelta = currentMousePosition - m_lastMousePosition;

	// if the delta movement is too big
	if (mouseDelta.length() > 50.0f)
	{
		m_lastMousePosition = currentMousePosition;
		return;
	}
	
	float horizontal_angel = -rotationSpeed * mouseDelta.x();
	float vertical_angel = -rotationSpeed * mouseDelta.y();
	m_strafeDirection = QVector3D::crossProduct(m_viewDirection, m_Up);
	QMatrix4x4 rotator;
	rotator.rotate(horizontal_angel, m_Up);
	rotator.rotate(vertical_angel, m_strafeDirection);
 	m_viewDirection = rotator * m_viewDirection;
	m_center = m_position + m_viewDirection;
	// replace the old position with the new one
	m_lastMousePosition = currentMousePosition;
}

void Camera::moveForward()
{
	m_position += movementSpeed * m_viewDirection;
}

void Camera::moveBackward()
{
	m_position -= movementSpeed * m_viewDirection;
}

void Camera::moveLeft()
{
	m_position -= movementSpeed * m_strafeDirection;
}

void Camera::moveRight()
{
	m_position += movementSpeed * m_strafeDirection;
}

void Camera::moveUp()
{
	m_position.setY(m_position.y() + movementSpeed);
}

void Camera::moveDown()
{
	m_position.setY(m_position.y() - movementSpeed);
}
