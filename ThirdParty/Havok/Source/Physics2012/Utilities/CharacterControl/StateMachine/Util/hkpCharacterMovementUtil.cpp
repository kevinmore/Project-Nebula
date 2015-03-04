/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/CharacterControl/StateMachine/Util/hkpCharacterMovementUtil.h>

void HK_CALL hkpCharacterMovementUtil::calculateMovement( const hkpMovementUtilInput& input, hkVector4& velocityOut )
{
	//
	// Move character relative to the surface we're standing on
	//

	// Construct a frame in world space
	hkRotation surfaceFrame;
	{
		hkVector4 binorm;
		binorm.setCross( input.m_forward, input.m_up  );


		if ( binorm.lengthSquared<3>().isLess( hkSimdReal_Eps ) )
		{
			// Bad configuration space
			return;
		}
		binorm.normalize<3>();

		hkVector4 tangent;
		tangent.setCross( binorm, input.m_surfaceNormal );
		tangent.normalize<3>();
		binorm.setCross( tangent, input.m_surfaceNormal );
		binorm.normalize<3>();

		surfaceFrame.setCols( tangent, binorm, input.m_surfaceNormal );
	}

	// Calculate the relative velocity in the surface Frame
	hkVector4 relative;
	{
		relative.setSub( input.m_currentVelocity, input.m_surfaceVelocity );
		relative.setRotatedInverseDir( surfaceFrame, relative );
	}

	// Calculate the difference between our desired and relative velocity
	hkVector4 diff;
	{
		if ( input.m_desiredVelocitySpace == hkpMovementUtilInput::SURFACE_SPACE )
		{
			diff.setSub( input.m_desiredVelocity, relative );
		}
		else // ( input.m_desiredVelocitySpace == WORLD_SPACE )
		{
			hkVector4 desiredVelocitySF;
			desiredVelocitySF.setRotatedInverseDir( surfaceFrame, input.m_desiredVelocity );
			diff.setSub( desiredVelocitySF, relative );
		}

		// Clamp it by maxVelocityDelta and limit it by gain.
		{
			const hkSimdReal maxVelDelta = hkSimdReal::fromFloat( input.m_maxVelocityDelta );
			const hkSimdReal inputGain = hkSimdReal::fromFloat( input.m_gain );
			const hkSimdReal len2 = diff.lengthSquared<3>();
			if ( len2 * inputGain * inputGain > maxVelDelta * maxVelDelta )
			{
				diff.mul( maxVelDelta * len2.sqrtInverse() );
			}
			else
			{
				diff.mul( inputGain );
			}
		}
	}
	
	relative.add( diff );

	// Transform back to world space and apply
	velocityOut.setRotatedDir( surfaceFrame, relative );

	// Add back in the surface velocity
	velocityOut.add( input.m_surfaceVelocity );
	HK_ASSERT( 0x447a0360,  velocityOut.isOk<3>() );
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
