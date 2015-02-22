#pragma once
#include <Utility/EngineCommon.h>
#include <Utility/Math.h>
#include <QSharedPointer>
#include <QElapsedTimer>
#include "Transform.h"
using namespace Math;

class Puppet;
class Component;
class IModel;
class Scene;

typedef QSharedPointer<Puppet> PuppetPtr;
typedef QSharedPointer<Component> ComponentPtr;
typedef QSharedPointer<IModel> ModelPtr;

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

	void setRotation(const quat& rotationQuaternion);
	void setRotation(const vec3& eulerAngles);
	void setRotation(double x, double y, double z);

	void setScale(const vec3& scale);
	void setScale(double x, double y, double z);
	void setScale(double scaleFactor);

	void setSpeed(const vec3& speed);
	void setSpeed(double x, double y, double z);

	vec3 globalSpeed() const;
	vec3 predictedPosition() const;

	void addPuppet(PuppetPtr p);
	void removePuppet(Puppet* p);
	QList<PuppetPtr> getPuppets();

	/////////////////////////////inline section///////////////////////////////////
	inline vec3 position() const { return m_transform.getPosition(); }
	inline vec3 rotation() const { return m_transform.getEulerAngles(); }
	inline vec3 scale() const { return m_transform.getScale(); }
	inline vec3 localSpeed() const { return m_speed; }
	inline const Transform& getTransform() const { return m_transform; }
	inline void setTransform(const Transform& trans) { m_transform = trans; }

	inline void setTransformMatrix(const mat4& transform)
	{
		m_modelMatrix = transform;
		m_modelMatrixDirty = false;
	}

	inline mat4 getTransformMatrix()
	{
		if(m_modelMatrixDirty)
		{
			m_modelMatrix = m_transform.getTransformMatrix();

			m_modelMatrixDirty = false;
		}
		GameObject* parent = dynamic_cast<GameObject*>(this->parent());
		if (parent)
			return parent->getTransformMatrix() * m_modelMatrix;
		else return m_modelMatrix;
	}

	/////////////////////////////inline section///////////////////////////////////

	void attachComponent(ComponentPtr pComponent);
	void detachComponent(ComponentPtr pComponent);
	QVector<ComponentPtr> getComponents();
	ComponentPtr getComponent(const QString& name);
	QStringList getComponentsTypes();

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
	void componentDetached(ComponentPtr comp);
	void transformChanged(const Transform& transform);

public slots:
	/// the 9 functions below will reset the model matrix
	void setFixedPositionX(double x);
	void setFixedPositionY(double y);
	void setFixedPositionZ(double z);

	void setFixedRotationX(double x);
	void setFixedRotationY(double y);
	void setFixedRotationZ(double z);

	void setFixedScaleX(double x);
	void setFixedScaleY(double y);
	void setFixedScaleZ(double z);

	/// the 12 functions below will NOT reset the model matrix
	void translateX(float x);
	void translateY(float y);
	void translateZ(float z);
	void translate(const vec3& delta);

	void rotateX(float x);
	void rotateY(float y);
	void rotateZ(float z);
	void rotate(const vec3& delta);
	void rotate(const quat& delta);

	void scaleX(float x);
	void scaleY(float y);
	void scaleZ(float z);
	void scale(const vec3& delta);

	void translateInWorld(const QString& paramString);
	void rotateInWorld(const QString& paramString);
	//void rotateInWorldAxisAndAngle(const QString& paramString);
	void setLocalSpeed(const QString& paramString);
	void resetSpeed();

	void calculateSpeed();

	void clearPuppets();

	void reset();

	void toggleFill(bool state);
	void toggleWireframe(bool state);
	void togglePoints(bool state);

private:
	Transform m_transform;
	vec3 m_prevPosition;

	vec3 m_speed;

	mat4 m_modelMatrix;

	mutable bool m_modelMatrixDirty;

	MovingBehaviour m_movingBehaviour;
	
	float m_time;
	QElapsedTimer m_lifeTimer;
	bool m_isMoving;
	QVector<ComponentPtr> m_components;
	QList<PuppetPtr> m_puppets;
	Scene* m_scene;
};

typedef QSharedPointer<GameObject> GameObjectPtr;