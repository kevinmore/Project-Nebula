#include "SceneCamera.h"
using namespace Math;

SceneCamera::SceneCamera(QObject *parent)
  : QObject(parent),
	m_position(Vector3D::UNIT_Z),
	m_upVector(Vector3D::UNIT_Y),
	m_viewCenter(Vector3D::ZERO),
	m_cameraToCenter(Vector3D::NEGATIVE_UNIT_Z),
	m_projectionType(PerspectiveProjection),
	m_nearPlane(0.1f),
	m_farPlane(1024.0f),
	m_fieldOfView(60.0f),
	m_aspectRatio(1.0f),
	m_left(-0.5f),
	m_right(0.5f),
	m_bottom(-0.5f),
	m_top(0.5f),
	m_viewMatrixDirty(true),
	m_viewProjectionMatrixDirty(true)
{
	updateOrthogonalProjection();
}

SceneCamera::~SceneCamera()
{
}

void SceneCamera::updateOrthogonalProjection()
{
	m_projectionMatrix.setToIdentity();
	m_projectionMatrix.ortho(m_left, m_right, m_bottom, m_top, m_nearPlane, m_farPlane);
	m_viewProjectionMatrixDirty = true;
}

void SceneCamera::setOrthographicProjection( float left, float right, float bottom, float top, float nearPlane, float farPlane )
{
	m_left = left;
	m_right = right;
	m_bottom = bottom;
	m_top = top;
	m_nearPlane = nearPlane;
	m_farPlane = farPlane;
	m_projectionType = OrthogonalProjection;

	updateOrthogonalProjection();
}


void SceneCamera::updatePerspectiveProjection()
{
	m_projectionMatrix.setToIdentity();
	m_projectionMatrix.perspective(m_fieldOfView, m_aspectRatio, m_nearPlane, m_farPlane);
	m_viewProjectionMatrixDirty = true;
}

void SceneCamera::setPerspectiveProjection( float fieldOfView, float aspect, float nearPlane, float farPlane )
{
	m_fieldOfView = fieldOfView;
	m_aspectRatio = aspect;
	m_nearPlane = nearPlane;
	m_farPlane = farPlane;
	m_projectionType = PerspectiveProjection;

	updatePerspectiveProjection();
}

SceneCamera::ProjectionType SceneCamera::projectionType() const
{
	return m_projectionType;
}

void SceneCamera::setProjectionType(ProjectionType type)
{
	m_projectionType = type;
}

QVector3D SceneCamera::position() const
{
	return m_position;
}

void SceneCamera::setPosition(const QVector3D& position)
{
	m_position = position;
	m_cameraToCenter = m_viewCenter - position;
	m_viewMatrixDirty = true;
}

void SceneCamera::setUpVector(const QVector3D& upVector)
{
	m_upVector = upVector;
	m_viewMatrixDirty = true;
}

QVector3D SceneCamera::upVector() const
{
	return m_upVector;
}

void SceneCamera::setViewCenter(const QVector3D& viewCenter)
{
	m_viewCenter = viewCenter;
	m_cameraToCenter = viewCenter - m_position;
	m_viewMatrixDirty = true;
}

QVector3D SceneCamera::viewCenter() const
{
	return m_viewCenter;
}

QVector3D SceneCamera::viewVector() const
{
	return m_cameraToCenter;
}

void SceneCamera::setNearPlane(const float& nearPlane)
{
	if(qFuzzyCompare(m_nearPlane, nearPlane))
		return;

	m_nearPlane = nearPlane;

	if(m_projectionType == PerspectiveProjection)
		updatePerspectiveProjection();
}

float SceneCamera::nearPlane() const
{
	return m_nearPlane;
}

void SceneCamera::setFarPlane(const float& farPlane)
{
	if(qFuzzyCompare(m_farPlane, farPlane))
		return;

	m_farPlane = farPlane;

	if(m_projectionType == PerspectiveProjection)
		updatePerspectiveProjection();
}

float SceneCamera::farPlane() const
{
	return m_farPlane;
}

void SceneCamera::setFieldOfView(const float& fieldOfView)
{
	if(qFuzzyCompare(m_fieldOfView, fieldOfView))
		return;

	m_fieldOfView = fieldOfView;

	if(m_projectionType == PerspectiveProjection)
		updatePerspectiveProjection();
}

float SceneCamera::fieldOfView() const
{
	return m_fieldOfView;
}

void SceneCamera::setAspectRatio(const float& aspectRatio)
{
	if(qFuzzyCompare(m_aspectRatio, aspectRatio))
		return;

	m_aspectRatio = aspectRatio;

	if(m_projectionType == PerspectiveProjection)
		updatePerspectiveProjection();
}

float SceneCamera::aspectRatio() const
{
	return m_aspectRatio;
}

void SceneCamera::setLeft(const float& left)
{
	if(qFuzzyCompare(m_left, left))
		return;

	m_left = left;

	if(m_projectionType == OrthogonalProjection)
		updateOrthogonalProjection();
}

float SceneCamera::left() const
{
	return m_left;
}

void SceneCamera::setRight(const float& right)
{
	if(qFuzzyCompare(m_right, right))
		return;

	m_right = right;

	if(m_projectionType == OrthogonalProjection)
		updateOrthogonalProjection();
}

float SceneCamera::right() const
{
	return m_right;
}

void SceneCamera::setBottom(const float& bottom)
{
	if(qFuzzyCompare(m_bottom, bottom))
		return;

	m_bottom = bottom;

	if(m_projectionType == OrthogonalProjection)
		updateOrthogonalProjection();
}

float SceneCamera::bottom() const
{
	return m_bottom;
}

void SceneCamera::setTop(const float& top)
{
	if (qFuzzyCompare(m_top, top))
		return;

	m_top = top;

	if(m_projectionType == OrthogonalProjection)
		updateOrthogonalProjection();
}

float SceneCamera::top() const
{
	return m_top;
}

QMatrix4x4 SceneCamera::viewMatrix() const
{
	if(m_viewMatrixDirty)
	{
		m_viewMatrix.setToIdentity();
		m_viewMatrix.lookAt(m_position, m_viewCenter, m_upVector);
		m_viewMatrixDirty = false;
	}

	return m_viewMatrix;
}

QMatrix4x4 SceneCamera::projectionMatrix() const
{
	return m_projectionMatrix;
}

QMatrix4x4 SceneCamera::viewProjectionMatrix() const
{
	if(m_viewMatrixDirty || m_viewProjectionMatrixDirty)
	{
		m_viewProjectionMatrix = m_projectionMatrix * viewMatrix();
		m_viewProjectionMatrixDirty = false;
	}

	return m_viewProjectionMatrix;
}

void SceneCamera::translate(const QVector3D& vLocal, CameraTranslationOption option)
{
	// Calculate the amount to move by in world coordinates
	QVector3D vWorld;

	if( ! qFuzzyIsNull(vLocal.x()) )
	{
		// Calculate the vector for the local x axis
		QVector3D x = QVector3D::crossProduct(m_cameraToCenter, m_upVector).normalized();
		vWorld += vLocal.x() * x;
	}

	if( ! qFuzzyIsNull( vLocal.y()) )
		vWorld += vLocal.y() * m_upVector;

	if( ! qFuzzyIsNull( vLocal.z()) )
		vWorld += vLocal.z() * m_cameraToCenter.normalized();

	// Update the camera position using the calculated world vector
	m_position += vWorld;

	// May be also update the view center coordinates
	if(option == TranslateViewCenter)
		m_viewCenter += vWorld;

	// Refresh the camera -> view center vector
	m_cameraToCenter = m_viewCenter - m_position;

	// Calculate a new up vector. We do this by:
	// 1) Calculate a new local x-direction vector from the cross product of the new
	//    camera to view center vector and the old up vector.
	// 2) The local x vector is the normal to the plane in which the new up vector
	//    must lay. So we can take the cross product of this normal and the new
	//    x vector. The new normal vector forms the last part of the orthonormal basis
	QVector3D x = QVector3D::crossProduct(m_cameraToCenter, m_upVector).normalized();
	m_upVector = QVector3D::crossProduct(x, m_cameraToCenter).normalized();

	m_viewMatrixDirty = true;
}

void SceneCamera::translateWorld(const QVector3D& vWorld , CameraTranslationOption option)
{
	// Update the camera position using the calculated world vector
	m_position += vWorld;

	// May be also update the view center coordinates
	if (option == TranslateViewCenter)
		m_viewCenter += vWorld;

	// Refresh the camera -> view center vector
	m_cameraToCenter = m_viewCenter - m_position;

	m_viewMatrixDirty = true;
}

QQuaternion SceneCamera::tiltRotation(const float& angle) const
{
	QVector3D xBasis = QVector3D::crossProduct(m_upVector, m_cameraToCenter.normalized()).normalized();

	return QQuaternion::fromAxisAndAngle(xBasis, -angle);
}

QQuaternion SceneCamera::panRotation(const float& angle) const
{
	return QQuaternion::fromAxisAndAngle(m_upVector, angle);
}

QQuaternion SceneCamera::panRotation(const float& angle, const QVector3D& axis) const
{
	return QQuaternion::fromAxisAndAngle(axis, angle);
}

QQuaternion SceneCamera::rollRotation(const float& angle) const
{
	return QQuaternion::fromAxisAndAngle(m_cameraToCenter, -angle);
}

void SceneCamera::tilt(const float& angle)
{
	QQuaternion q = tiltRotation(angle);
	rotate(q);
}

void SceneCamera::pan(const float& angle)
{
	QQuaternion q = panRotation(-angle);
	rotate(q);
}

void SceneCamera::pan(const float& angle, const QVector3D& axis)
{
	QQuaternion q = panRotation(-angle, axis);
	rotate(q);
}

void SceneCamera::roll(const float& angle)
{
	QQuaternion q = rollRotation(-angle);
	rotate(q);
}

void SceneCamera::tiltAboutViewCenter(const float& angle)
{
	QQuaternion q = tiltRotation(-angle);
	rotateAboutViewCenter(q);
}

void SceneCamera::panAboutViewCenter(const float& angle)
{
	QQuaternion q = panRotation(angle);
	rotateAboutViewCenter(q);
}

void SceneCamera::rollAboutViewCenter(const float& angle)
{
	QQuaternion q = rollRotation(angle);
	rotateAboutViewCenter(q);
}

void SceneCamera::rotate(const QQuaternion& q)
{
	m_upVector = q.rotatedVector(m_upVector);
	m_cameraToCenter = q.rotatedVector(m_cameraToCenter);
	m_viewCenter = m_position + m_cameraToCenter;

	m_viewMatrixDirty = true;
}

void SceneCamera::rotateAboutViewCenter(const QQuaternion& q)
{
	m_upVector = q.rotatedVector(m_upVector);
	m_cameraToCenter = q.rotatedVector(m_cameraToCenter);
	m_position = m_viewCenter - m_cameraToCenter;

	m_viewMatrixDirty = true;
}

void SceneCamera::resetCamera()
{
// 	m_position = QVector3D(Vector3D::UNIT_Z);
// 	m_upVector = QVector3D(Vector3D::UNIT_Y);
// 	m_viewCenter = QVector3D(Vector3D::ZERO);
// 	m_cameraToCenter = QVector3D(Vector3D::NEGATIVE_UNIT_Z);
	m_position = QVector3D(0.0f, 6.0f, 6.0f);
	m_upVector = QVector3D(0.0f, 3.6f, 0.0f);
	m_viewCenter = QVector3D(0.0f, 1.0f, 0.0f);
	m_cameraToCenter = QVector3D(Vector3D::NEGATIVE_UNIT_Z);
	m_viewMatrixDirty = true;
}
