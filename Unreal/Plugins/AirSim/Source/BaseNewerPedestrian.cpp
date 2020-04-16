// Fill out your copyright notice in the Description page of Project Settings.

#include "BaseNewerPedestrian.h"
#include "Kismet/KismetMathLibrary.h"


// Sets default values
ABaseNewerPedestrian::ABaseNewerPedestrian()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	SkeletalMesh = CreateDefaultSubobject<USkeletalMeshComponent>(TEXT("SkeletalMesh"));
	SetRootComponent(SkeletalMesh);
	//CapsuleComponent = CreateDefaultSubobject<UCapsuleComponent>(TEXT("CapsuleComponent"));

	desired_speed = 0;

}

// Called when the game starts or when spawned
void ABaseNewerPedestrian::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ABaseNewerPedestrian::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (desired_speed > 0)
	{
		GetActorRotation();
		FRotator TargetRot = FRotationMatrix::MakeFromX(current_goal_position - GetActorLocation()).Rotator();

		// rotate until we're facing target, then start moving
		// ignore roll, that will make us rotate into the ground
		
		if (!FMath::IsNearlyEqual(GetActorRotation().Yaw, TargetRot.Yaw, 0.1f) || !FMath::IsNearlyEqual(GetActorRotation().Pitch, TargetRot.Pitch, 0.1f))
		{
																								// rotation speed 
			SetActorRotation(FMath::RInterpConstantTo(GetActorRotation(), TargetRot, DeltaTime, 100.0f));
		}
		else
		{
			float move_speed_ = 100.0f;
			if (desired_speed == 0)
			{
				move_speed_ = 0.0f;
			}
			else if (desired_speed == 1)
			{
				// walking speed
				move_speed_ = 100.0f;
			}
			else if (desired_speed == 2)
			{
				// speed walking
				move_speed_ = 150.0f;
			}
			else if (desired_speed == 3)
			{
				// running speed
				move_speed_ = 200.0f;
			}
			if (FVector::PointsAreNear(GetActorLocation(), current_goal_position, 5.0f))
			{
				bIsMoving = false;
				desired_speed = 0;
			}
			else
			{
				bIsMoving = true;
				SetActorLocation(FMath::VInterpConstantTo(GetActorLocation(), current_goal_position, DeltaTime, move_speed_));
			}
		}
	}

}


void ABaseNewerPedestrian::move(FVector goal_position, int32 speed)
{
	current_goal_position = goal_position;
	desired_speed = speed;
}

void ABaseNewerPedestrian::stop()
{
	desired_speed = 0;
}
int32 ABaseNewerPedestrian::GetSpeed() const
{
	return desired_speed;
}
bool ABaseNewerPedestrian::GetIsMoving() const
{
	return bIsMoving;
}