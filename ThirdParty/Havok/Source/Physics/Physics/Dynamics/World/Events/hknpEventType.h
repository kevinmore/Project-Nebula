/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_EVENT_TYPE_H
#define HKNP_EVENT_TYPE_H


/// Event type
struct hknpEventType
{
	enum Enum
	{
		//
		// Binary events
		//

		MANIFOLD_STATUS,			///< manifold was created/deleted/deactivated or activated, fired if hknpBody::m_flags |= hknpBody::RAISE_MANIFOLD_STATUS_EVENTS
		MANIFOLD_PROCESSED,			///< manifold was processed, fired if hknpBody::m_flags |= hknpBody::RAISE_MODIFIER_PROCESS_EVENTS
		CONTACT_IMPULSE,			///< manifold started/continued/finished applying non-zero contact impulses.
		CONTACT_IMPULSE_CLIPPED,	///< manifold impulse was clipped to the specified limit, e.g. hknpMaterial::m_maxContactImpulse
		CONSTRAINT_FORCE,			///< constraint force, will be fired if you set hknpBody::m_flags |= hknpBody::RAISE_CONSTRAINT_FORCE_EVENTS
		CONSTRAINT_FORCE_EXCEEDED,	///< constraint force exceeded, requires a suitable modifier.
		TRIGGER_VOLUME,				///< information about a trigger volume, fired if hknpMaterial::m_triggerVolumeMode is set.

		USER_BINARY_0,				///< User event.
		USER_BINARY_1,
		USER_BINARY_2,
		USER_BINARY_3,

		END_OF_BINARY_BODY_EVENT,	// internal end marker of all binary events

		//
		// Unary events
		//

		BODY_ACTIVATION,			///< a body was activated or deactivated
		BODY_EXITED_BROAD_PHASE,	///< A body has left the broad phase.
		LINEAR_INTEGRATION_CLIPPED,	///< Called when linear velocity is not fully integrated because of CCD reasons.

		USER_UNARY_0,				///< User event.
		USER_UNARY_1,
		USER_UNARY_2,
		USER_UNARY_3,

		END_OF_UNARY_BODY_EVENTS,	// internal end marker of all unary events

		//
		// Other events
		//

		RESERVED_0,
		RESERVED_1,
		RESERVED_2,
		RESERVED_3,
	};
};

#endif // HKNP_EVENT_TYPE_H

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
