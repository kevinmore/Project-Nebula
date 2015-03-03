/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_ACTION_H
#define HKNP_ACTION_H

#include <Physics/Physics/hknpTypes.h>

class hknpCdPairWriter;
class hknpSimulationThreadContext;
struct hknpSolverInfo;


/// This is the base class from which user actions are derived.
/// Actions basically equal an applyAction() callback that are called for a group of active bodies.
/// The action can decide to deactivate itself if all bodies involved are ready for deactivation.
/// Warning: This feature may change significantly in future versions.
class hknpAction : public hkReferencedObject
{
	public:

		/// The result of the applyAction() method.
		enum ApplyActionResult
		{
			RESULT_OK,			///< OK return, keep going.
			RESULT_DEACTIVATE,  ///< The action requests to be deactivated.
			RESULT_REMOVE,		///< The action wants itself to be removed from the physics.
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor.
		HK_FORCE_INLINE hknpAction( hkUlong userData = 0 );

		/// Serializing constructor.
		HK_FORCE_INLINE hknpAction( class hkFinishLoadedObjectFlag flag );

		/// Apply the action.
		///   - The action (and only the action) is allowed to access and change the motions directly.
		///   - The action is allowed to perform asynchronous collision queries (like raycast).
		///	  - The action is not allowed to apply forces on other bodies than the one owned by the action.
		///     If you need to you can use the hknpSetPointVelocityCommand().
		///   - The action is required to send all links between the bodies to the pairWriter.
		virtual ApplyActionResult applyAction(
			const hknpSimulationThreadContext& tl, const hknpSolverInfo& stepInfo,
			hknpCdPairWriter* HK_RESTRICT pairWriter ) = 0;

		/// Get write access to a motion.
		/// Can be called by hknpAction::apply() implementation.
		HK_FORCE_INLINE hknpMotion* getMotion( hknpWorld* world, const hknpBody& body );

		/// Get the bodies the action uses.
		/// \a bodiesOut must have at least 16 elements reserved.
		virtual void getBodies( hkArray<hknpBodyId>* HK_RESTRICT bodiesOut ) const = 0;

		/// This internal helper method should be used by an hknpAction::apply() implementation to tell the
		/// deactivation system that two bodies cannot deactivate at different times.
		HK_FORCE_INLINE void addLink(
			const hknpMotion* motionA, const hknpMotion* motionB, hknpCdPairWriter* HK_RESTRICT pairWriter );

		/// Internal helper function to add a link to the pair writer. This silently assumes that the input motions are not static.
		void addLinkUnchecked(
			const hknpMotion* motionA, const hknpMotion* motionB, hknpCdPairWriter* HK_RESTRICT pairWriter );

		/// Called when the world coordinate system has been shifted by adding 'offset' to all positions.
		virtual void onShiftWorld( hkVector4Parameter offset ) = 0;

	public:

		/// User data.
		hkUlong m_userData;
};


/// Base class for all actions that are working on a single body.
class hknpUnaryAction : public hknpAction
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor.
		HK_FORCE_INLINE hknpUnaryAction( hknpBodyId idA, hkUlong userData = 0 );

		/// Serializing constructor.
		HK_FORCE_INLINE hknpUnaryAction( class hkFinishLoadedObjectFlag flag );

		//
		// hknpAction implementation
		//

		virtual void getBodies( hkArray<hknpBodyId>* HK_RESTRICT bodiesOut ) const;

	protected:

		/// Return body (if valid and active) or the action's desired state.
		HK_FORCE_INLINE ApplyActionResult getAndCheckBodies( hknpWorld* world, const hknpBody*& bodyOut );

	public:

		/// The body
		hknpBodyId m_body;	//+overridetype(hkUint32)
};


/// Base class for all actions that are working on a pair of bodies.
class hknpBinaryAction : public hknpAction
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor.
		HK_FORCE_INLINE hknpBinaryAction( hknpBodyId idA, hknpBodyId idB = hknpBodyId(0), hkUlong userData = 0 );

		/// Serializing constructor.
		HK_FORCE_INLINE hknpBinaryAction( class hkFinishLoadedObjectFlag flag );

		/// Action initialization method
		virtual void initialize( hknpBodyId idA, hknpBodyId idB = hknpBodyId(0), hkUlong userData = 0 );

		//
		// hknpAction implementation
		//

		virtual void getBodies( hkArray<hknpBodyId>* HK_RESTRICT bodiesOut ) const;

	protected:

		/// Return body (if valid and active) or the action's desired state.
		HK_FORCE_INLINE ApplyActionResult getAndCheckBodies(
			hknpWorld* world, const hknpBody*& bodyAOut, const hknpBody*& bodyBOut );

	public:

		/// Body A or 0 for fixed bodies.
		hknpBodyId m_bodyA;	//+overridetype(hkUint32)

		/// Body B or 0 for fixed bodies.
		hknpBodyId m_bodyB;	//+overridetype(hkUint32)
};


#include <Physics/Physics/Dynamics/Action/hknpAction.inl>


#endif // HKNP_ACTION_H

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
