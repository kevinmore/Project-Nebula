/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

#include <Physics2012/Collide/Query/CastUtil/hkpLinearCastInput.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>

//this
#include <Physics2012/Utilities/Weapons/hkpFirstPersonGun.h>


hkpFirstPersonGun::hkpFirstPersonGun()
:	m_type(WEAPON_TYPE_INVALID)
,	m_name("")
,	m_keyboardKey(KEY_F2)
{
	HK_COMPILE_TIME_ASSERT(WEAPON_TYPE_NUM_TYPES <= (1 << (sizeof(m_type) * 8)) );
}

const char* hkpFirstPersonGun::getName() const
{ 
	return m_name; 
}

void hkpFirstPersonGun::quitGun( hkpWorld* world )
{ 
	reset(world); 
}

/*static*/ hkResult HK_CALL hkpFirstPersonGun::sweepSphere(const hkpWorld* world, const hkVector4& sweepStart, hkReal radius, const hkVector4& sweepEnd, SweepSphereOut& out )
{
	// do linear sweep

	//Create casting object
	hkpSphereShape* sphereShape = new hkpSphereShape( radius );
	hkTransform startTransform; startTransform.setIdentity();
	startTransform.getTranslation() = sweepStart ;

	hkpCollidable collidable( sphereShape, &startTransform );

	hkpClosestCdPointCollector castCollector;
	{
		hkpLinearCastInput linearCastInput;
		linearCastInput.m_to = sweepEnd;
		world->linearCast( &collidable, linearCastInput, castCollector);
	}

	sphereShape->removeReference();

	if (castCollector.hasHit())
	{
		out.m_contactPoint = castCollector.getHitContact();
		hkpRigidBody* hitEntity = hkpGetRigidBody( castCollector.getHit().m_rootCollidableB );
		out.m_body = hitEntity;
		out.m_shapeKey = castCollector.getHit().m_shapeKeyB;
		return HK_SUCCESS;
	}
	else
	{

		out.m_body = HK_NULL;
		out.m_contactPoint.setPosition( sweepEnd );
		out.m_contactPoint.setSeparatingNormal( hkVector4::getZero() );
		out.m_shapeKey = HK_INVALID_SHAPE_KEY;
		return HK_FAILURE;
	}
}

/*static*/ void HK_CALL hkpFirstPersonGun::calcVelocityToTarget(const hkVector4& position, const hkVector4& target, const hkVector4& gravity, hkReal speedR, hkVector4& velocity)
{
	hkVector4 dist; dist.setSub(target, position); 
	hkSimdReal distLen = dist.length<3>();
	const hkSimdReal speed = hkSimdReal::fromFloat(speedR);
	if (distLen > hkSimdReal_Eps)
	{
		hkSimdReal time = distLen / speed;
		hkVector4 extraVelocity; extraVelocity.setMul(hkSimdReal_Inv2 * (-time), gravity);
		hkSimdReal extraVelLen = extraVelocity.length<3>();
		if (extraVelLen > speed)
		{
			// clip extra velocity .. to 45degree deflection
			extraVelocity.mul(speed / extraVelLen);
		}
		velocity.setAddMul(extraVelocity, dist, speed / distLen);
	}
	else
	{
		// Ignore:
		velocity = dist;
	}
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
