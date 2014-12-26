#pragma once
#include <Utility/DataTypes.h>
#include <Utility/Math.h>
#include <Scene/GameObject.h>

class Camera : public QObject
{
	Q_OBJECT

public:
	Camera(GameObject* followingTarget = 0, QObject *parent = 0);
	~Camera();

	enum ProjectionType
	{
		Orthogonal,
		Perspective
	};

	enum ViewType
	{
		FirstPerson,
		ThirdPerson
	};

	enum CameraTranslationOption
	{
		TranslateViewCenter,
		DontTranslateViewCenter
	};

	QVector3D position()   const;
	QVector3D upVector()   const;
	QVector3D viewCenter() const;
	QVector3D viewVector() const;

	ProjectionType projectionType() const;
	void setProjectionType(ProjectionType type);
	
	float left()   const;
	float right()  const;
	float bottom() const;
	float top()    const;
	void setLeft(const float& left);
	void setRight(const float& right);
	void setBottom(const float& bottom);
	void setTop(const float& top);

	float nearPlane()   const;
	float farPlane()    const;
	float fieldOfView() const;
	float aspectRatio() const;
	void setNearPlane(const float& nearPlane);
	void setFarPlane(const float& nearPlane);
	void setFieldOfView(const float& fieldOfView);
	void setAspectRatio(const float& aspectRatio);
	
	void updateOrthogonalProjection();
	void setOrthographicProjection(float left, float right,	float bottom, float top, float nearPlane, float farPlane);

	void updatePerspectiveProjection();
	void setPerspectiveProjection(float fieldOfView, float aspect, float nearPlane, float farPlane);

	QMatrix4x4 viewMatrix() const;
	QMatrix4x4 projectionMatrix() const;
	QMatrix4x4 viewProjectionMatrix() const;

	QQuaternion tiltRotation(const float& angle) const;
	QQuaternion rollRotation(const float& angle) const;

	QQuaternion panRotation(const float& angle) const;
	QQuaternion panRotation(const float& angle, const QVector3D& axis) const;


	void setPosition(const QVector3D& position);
	void setUpVector(const QVector3D& upVector);
	void setViewCenter(const QVector3D& viewCenter);

	// Translate relative to camera orientation axes
	void translate(const QVector3D& vLocal, CameraTranslationOption option = TranslateViewCenter);

	// Translate relative to world axes
	void translateWorld(const QVector3D& vWorld, CameraTranslationOption option = TranslateViewCenter);

	void tilt(const float& angle);
	void roll(const float& angle);

	void pan(const float& angle);
	void pan(const float& angle, const QVector3D& axis);

	void rollAboutViewCenter(const float& angle);
	void tiltAboutViewCenter(const float& angle);
	void panAboutViewCenter(const float& angle);

	void rotate(const QQuaternion& q);
	void rotateAboutViewCenter(const QQuaternion& q);

	// make the camera follow a game object
	void followTarget(GameObject* target);

	vec3 getViewDirection() { return m_viewDirection; }
	bool isViewCenterFixed() { return m_viewCenterFixed; }

	// camera movement
	inline void setSideSpeed(float vx)     { m_viewDirection.setX(vx); }
	inline void setVerticalSpeed(float vy) { m_viewDirection.setY(vy); }
	inline void setForwardSpeed(float vz)  { m_viewDirection.setZ(vz); }
	inline void setViewCenterFixed(bool b) { m_viewCenterFixed = b; }

	// camera movement rotation
	inline void setPanAngle(float angle)  { m_panAngle  = angle; }
	inline void setTiltAngle(float angle) { m_tiltAngle = angle; }

	void update(const float currentTime);

public slots:
	void resetCamera();
	void releaseTarget();
	void switchToFirstPersonCamera(bool status);
	void switchToThirdPersonCamera(bool status);

private:
	QVector3D m_position;
	QVector3D m_upVector;
	QVector3D m_viewCenter;
	QVector3D m_cameraToCenter;

	ProjectionType m_projectionType;
	ViewType m_viewType;

	float m_nearPlane;
	float m_farPlane;
	float m_fieldOfView;
	float m_aspectRatio;

	float m_left;
	float m_right;
	float m_bottom;
	float m_top;

	mutable QMatrix4x4 m_viewMatrix;
	mutable QMatrix4x4 m_projectionMatrix;
	mutable QMatrix4x4 m_viewProjectionMatrix;

	mutable bool m_viewMatrixDirty;
	mutable bool m_viewProjectionMatrixDirty;

	// instantiate variables
	QVector3D m_viewDirection;
	bool m_viewCenterFixed;
	bool m_isFollowing;
	GameObject* m_followingTarget;

	float m_panAngle;
	float m_tiltAngle;

	float m_time;
	const float m_metersToUnits;
};

