/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CHARACTER_SUPPORT_H
#define HKNP_CHARACTER_SUPPORT_H

#include <Common/Base/hkBase.h>

/// Surface information returned from character checkSupport() queries.
struct hknpCharacterSurfaceInfo
{
	public:

		/// The supported state of the character.
		enum SupportedState
		{
			/// This state implies there are no surfaces underneath the character.
			UNSUPPORTED = 0,

			/// This state means that there are surfaces underneath the character, but they are too
			/// steep to prevent the character sliding downwards.
			SLIDING = 1,

			/// This state means the character is supported, and will not slide.
			SUPPORTED = 2
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCharacterSurfaceInfo );

		/// Constructor. Initializes all members.
		HK_FORCE_INLINE hknpCharacterSurfaceInfo();

		/// Constructor.
		HK_FORCE_INLINE hknpCharacterSurfaceInfo(
			hkVector4Parameter up, hkVector4Parameter velocity = hkVector4::getZero(),
			const SupportedState state = SUPPORTED, hkBool isDynamic = false );

		/// Copy from another hknpCharacterSurfaceInfo.
		HK_FORCE_INLINE void set( const hknpCharacterSurfaceInfo& other );

		/// Check if the numeric data members have valid values.
		HK_FORCE_INLINE hkBool isValid() const;

	public:

		/// Is the surface dynamic (i.e., not fixed/static or keyframed).
		hkBool m_isSurfaceDynamic;

		/// The supported state of the character.
		hkEnum<SupportedState,hkUint8> m_supportedState;

		/// The excess distance to the surface, which the controller should try to reduce this by applying gravity.
		/// This behavior is not required by the proxy character controller, so it sets the surfaceDistance to 0.
		hkReal m_surfaceDistanceExcess;

		/// The average surface normal in this given direction.
		hkVector4 m_surfaceNormal;

		/// The average surface velocity.
		hkVector4 m_surfaceVelocity;
};

#include <Physics/Physics/Extensions/CharacterControl/hknpCharacterSurfaceInfo.inl>

#endif // HKNP_CHARACTER_SUPPORT_H

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
