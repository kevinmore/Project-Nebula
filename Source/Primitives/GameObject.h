#pragma once
#include <Utility/EngineCommon.h>
#include <Utility/Math.h>
#include <QSharedPointer>
#include <QElapsedTimer>
using namespace Math;

class Puppet;
class Component;
class AbstractModel;
class Scene;

typedef QSharedPointer<Puppet> PuppetPtr;
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
	inline vec3 position() const { return m_position; }
	inline vec3 rotation() const { return m_rotation; }
	inline vec3 scale() const { return m_scale; }
	inline vec3 localSpeed() const { return m_speed; }

	inline void setTransformMatrix(const mat4& transform)
	{
		m_modelMatrix = transform;
		m_modelMatrixDirty = false;
	}

	inline mat4 getTransformMatrix()
	{
		if(m_modelMatrixDirty)
		{
			m_modelMatrix.setToIdentity();

			m_modelMatrix.translate(m_position);
			m_modelMatrix.rotate(m_rotation.x(), Vector3::UNIT_X);
			m_modelMatrix.rotate(m_rotation.y(), Vector3::UNIT_Y);
			m_modelMatrix.rotate(m_rotation.z(), Vector3::UNIT_Z);
			m_modelMatrix.scale(m_scale);

			m_modelMatrixDirty = false;
		}
		emit updateTransformation(m_position, m_rotation, m_scale);
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
	void updateTransformation(const vec3& pos, const vec3& rot, const vec3& scale);

public slots:
	/// the 9 functions below will reset the model matrix
	void fixedTranslateX(double x);
	void fixedTranslateY(double y);
	void fixedTranslateZ(double z);

	void fixedRotateX(double x);
	void fixedRotateY(double y);
	void fixedRotateZ(double z);

	void fixedScaleX(double x);
	void fixedScaleY(double y);
	void fixedScaleZ(double z);

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
	void rotateInWorldAxisAndAngle(const QString& paramString);
	void setLocalSpeed(const QString& paramString);
	void resetSpeed();

	void calculateSpeed();

	void clearPuppets();

	void reset();

	void toggleFill(bool state);
	void toggleWireframe(bool state);
	void togglePoints(bool state);

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
	QList<PuppetPtr> m_puppets;
	Scene* m_scene;
};

typedef QSharedPointer<GameObject> GameObjectPtr;