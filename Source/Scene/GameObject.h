#pragma once
#include <Utility/EngineCommon.h>
#include <Utility/Math.h>
#include <QSharedPointer>
#include <QElapsedTimer>
using namespace Math;

class Component;
class AbstractModel;
class Scene;

typedef QSharedPointer<Component> ComponentPtr;
typedef QSharedPointer<AbstractModel> ModelPtr;

class GameObject : public QObject
{
	Q_OBJECT
	Q_PROPERTY(bool moving READ isMoving WRITE setMoving)
	Q_PROPERTY(vec3 localSpeed READ localSpeed WRITE setSpeed)

public:
	GameObject(Scene* scene, GameObject* parent = 0);
	~GameObject();

	bool isMoving() const;
	void setMoving(bool status);

	void setPosition(const vec3& positionVector);
	void setPosition(double x, double y, double z);
	void translateInWorld(const vec3& delta);

	void setRotation(const vec3& rotationVector);
	void setRotation(double x, double y, double z);
	void rotateInWorld(const QQuaternion& delta);

	void setScale(const vec3& scale);
	void setScale(double x, double y, double z);
	void setScale(double scaleFactor);

	void setSpeed(const vec3& speed);
	void setSpeed(double x, double y, double z);

	vec3 globalSpeed() const;
	vec3 predictedPosition() const;

	/////////////////////////////inline section///////////////////////////////////
	inline vec3 position() const { return m_position; }
	inline vec3 rotation() const { return m_rotation; }
	inline vec3 scale() const { return m_scale; }
	inline vec3 localSpeed() const { return m_speed; }

	inline void setTransformMatrix(const mat4& transform)
	{
		m_modelMatrix = transform;
		m_modelMatrixDirty = false;
	}

	inline mat4 modelMatrix()
	{
		if(m_modelMatrixDirty)
		{
			m_modelMatrix.setToIdentity();

			m_modelMatrix.translate(m_position);
			m_modelMatrix.rotate(m_rotation.x(), Vector3D::UNIT_X);
			m_modelMatrix.rotate(m_rotation.y(), Vector3D::UNIT_Y);
			m_modelMatrix.rotate(m_rotation.z(), Vector3D::UNIT_Z);
			m_modelMatrix.scale(m_scale);

			m_modelMatrixDirty = false;
		}
		GameObject* parent = dynamic_cast<GameObject*>(this->parent());
		if (parent)
			return parent->modelMatrix() * m_modelMatrix;
		else return m_modelMatrix;
	}

	/////////////////////////////inline section///////////////////////////////////

	void attachComponent(ComponentPtr pComponent);
	QVector<ComponentPtr> getComponents();
	ComponentPtr getComponent(const QString& name);

	enum MovingBehaviour
	{
		CONSECUTIVE,
		DISCRETE
	};

	void setMovingBehaviour(MovingBehaviour type);
	MovingBehaviour movingBehaviour() const;

	void setScene(Scene* scene);
	Scene* getScene() const;

signals:
	void synchronized();
	void componentAttached(ComponentPtr comp);

public slots:
	void translateX(double x);
	void translateY(double y);
	void translateZ(double z);

	void rotateX(double x);
	void rotateY(double y);
	void rotateZ(double z);

	void scaleX(double x);
	void scaleY(double y);
	void scaleZ(double z);

	void translateInWorld(const QString& paramString);
	void rotateInWorld(const QString& paramString);
	void rotateInWorldAxisAndAngle(const QString& paramString);
	void setLocalSpeed(const QString& paramString);
	void resetSpeed();

	void calculateSpeed();

	void reset();

private:
	vec3 m_position, m_prevPosition;
	vec3 m_rotation;
	vec3 m_scale;
	vec3 m_speed;

	mat4 m_modelMatrix;

	bool m_modelMatrixDirty;

	MovingBehaviour m_movingBehaviour;
	
	float m_time;
	QElapsedTimer m_lifeTimer;
	bool m_isMoving;
	QVector<ComponentPtr> m_components;

	Scene* m_scene;
};

typedef QSharedPointer<GameObject> GameObjectPtr;