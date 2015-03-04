/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_CONTACT_JACOBIAN_UTIL_H
#define HKNP_CONTACT_JACOBIAN_UTIL_H


/// Utility functions for modifying contact Jacobians.
namespace hknpContactJacobianUtil
{
	/// Disable the contact points.
	void HK_CALL disableContacts( hknpMxContactJacobian* contactJacobian, int manifoldIndex );

	/// Set the maximum impulse of the contact manifold (per solver substep).
	void HK_CALL setMaxImpulse( hknpMxContactJacobian* contactJacobian, int manifoldIndex, hkReal impulse );

	/// Set the maximum friction impulse of the contact manifold (per solver substep).
	void HK_CALL setMaxFrictionImpulse( hknpMxContactJacobian* contactJacobian, int manifoldIndex, hkReal friction );

	/// Scale the effective mass of the contact points.
	void HK_CALL scaleEffectiveMass( hknpMxContactJacobian* contactJacobian, int manifoldIndex, hkVector4Parameter scale );

	/// Scale the contact point distance if it is negative.
	void HK_CALL scalePenetrations( hknpMxContactJacobian* contactJacobian, int manifoldIndex, hkVector4Parameter scale );

	/// Reduce rotation effects of one body.
	/// Scale is used for the contact points, frictionScale is used for the friction forces.
	void HK_CALL scaleAngularPart(
		hknpMxContactJacobian* contactJacobian, int manifoldIndex, hknpBodyId bodyId,
		hkVector4Parameter scale, hkReal frictionScale );

	/// Set a virtual surface velocity. Affects friction only.
	void HK_CALL setSurfaceVelocity(
		const hknpSolverInfo& solverInfo,
		hknpManifoldCollisionCache* cache, hknpMxContactJacobian* contactJacobian, int mxJacIdx,
		hkVector4Parameter linearVelocity,	hkVector4Parameter angularVelocity );
}


#endif // HKNP_CONTACT_JACOBIAN_UTIL_H

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
